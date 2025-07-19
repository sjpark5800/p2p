# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, Stream

from camel.configs import DEEPSEEK_API_PARAMS, DeepSeekConfig
from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import BaseTokenCounter, OpenAITokenCounter, api_keys_required

logger = get_logger(__name__)

class DeepInfraDeepSeekModel(BaseModelBackend):
    r"""DeepSeek API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`DeepSeekConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the DeepSeek service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the DeepSeek service.
            (default: :obj:`https://api.deepseek.com`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter`
            will be used. (default: :obj:`None`)

    References:
        https://api-docs.deepseek.com/
    """

    @api_keys_required(
        [
            ("api_key", "DEEPINFRA_API_KEY"),
        ]
    )
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = DeepSeekConfig().as_dict()
        api_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
        url = url or os.environ.get(
            "DEEPSEEK_API_BASE_URL",
            "https://api.deepinfra.com/v1/openai",
        )
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )

        self._client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(
                model=ModelType.GPT_4O_MINI
            )
        return self._token_counter

    def run(
        self,
        messages: List[OpenAIMessage],
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of DeepSeek chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        # deepseek reasoner has limitations
        # reference: https://api-docs.deepseek.com/guides/reasoning_model#api-parameters
        if self.model_type in [
            ModelType.DEEPSEEK_REASONER,
        ]:
            import re

            logger.warning(
                "You are using a DeepSeek Reasoner model, "
                "which has certain limitations, reference: "
                "`https://api-docs.deepseek.com/guides/reasoning_model#api-parameters`"
            )

            # Check and remove unsupported parameters and reset the fixed
            # parameters
            unsupported_keys = [
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "logprobs",
                "top_logprobs",
                "tools",
            ]
            for key in unsupported_keys:
                if key in self.model_config_dict:
                    del self.model_config_dict[key]

            # Remove thinking content from messages before sending to API
            # This ensures only the final response is sent, excluding
            # intermediate thought processes
            messages = [
                {  # type: ignore[misc]
                    **msg,
                    'content': re.sub(
                        r'<think>.*?</think>',
                        '',
                        msg['content'],  # type: ignore[arg-type]
                        flags=re.DOTALL,
                    ).strip(),
                }
                for msg in messages
            ]

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **self.model_config_dict,
        )

        # Handle reasoning content with <think> tags at the beginning
        if (
            self.model_type
            in [
                ModelType.DEEPSEEK_REASONER,
            ]
            and os.environ.get("GET_REASONING_CONTENT", "false").lower()
            == "true"
        ):
            reasoning_content = response.choices[0].message.reasoning_content
            combined_content = (
                f"<think>\n{reasoning_content}\n</think>\n"
                if reasoning_content
                else ""
            ) + response.choices[0].message.content

            response = ChatCompletion.construct(
                id=response.id,
                choices=[
                    dict(
                        index=response.choices[0].index,
                        message={
                            "role": response.choices[0].message.role,
                            "content": combined_content,
                            "tool_calls": None,
                        },
                        finish_reason=response.choices[0].finish_reason
                        if response.choices[0].finish_reason
                        else None,
                    )
                ],
                created=response.created,
                model=response.model,
                object="chat.completion",
                usage=response.usage,
            )

        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to DeepSeek API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to DeepSeek API.
        """
        for param in self.model_config_dict:
            if param not in DEEPSEEK_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into DeepSeek model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get("stream", False)


class DeepSeekModel(BaseModelBackend):
    r"""DeepSeek API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`DeepSeekConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the DeepSeek service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the DeepSeek service.
            (default: :obj:`https://api.deepseek.com`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter`
            will be used. (default: :obj:`None`)

    References:
        https://api-docs.deepseek.com/
    """

    @api_keys_required(
        [
            ("api_key", "DEEPSEEK_API_KEY"),
        ]
    )
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = DeepSeekConfig().as_dict()
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        url = url or os.environ.get(
            "DEEPSEEK_API_BASE_URL",
            "https://api.deepseek.com",
        )
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )

        self._client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(
                model=ModelType.GPT_4O_MINI
            )
        return self._token_counter

    def run(
        self,
        messages: List[OpenAIMessage],
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of DeepSeek chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        # deepseek reasoner has limitations
        # reference: https://api-docs.deepseek.com/guides/reasoning_model#api-parameters
        if self.model_type in [
            ModelType.DEEPSEEK_REASONER,
        ]:
            import re

            logger.warning(
                "You are using a DeepSeek Reasoner model, "
                "which has certain limitations, reference: "
                "`https://api-docs.deepseek.com/guides/reasoning_model#api-parameters`"
            )

            # Check and remove unsupported parameters and reset the fixed
            # parameters
            unsupported_keys = [
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "logprobs",
                "top_logprobs",
                "tools",
            ]
            for key in unsupported_keys:
                if key in self.model_config_dict:
                    del self.model_config_dict[key]

            # Remove thinking content from messages before sending to API
            # This ensures only the final response is sent, excluding
            # intermediate thought processes
            messages = [
                {  # type: ignore[misc]
                    **msg,
                    'content': re.sub(
                        r'<think>.*?</think>',
                        '',
                        msg['content'],  # type: ignore[arg-type]
                        flags=re.DOTALL,
                    ).strip(),
                }
                for msg in messages
            ]

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **self.model_config_dict,
        )

        # Handle reasoning content with <think> tags at the beginning
        if (
            self.model_type
            in [
                ModelType.DEEPSEEK_REASONER,
            ]
            and os.environ.get("GET_REASONING_CONTENT", "false").lower()
            == "true"
        ):
            reasoning_content = response.choices[0].message.reasoning_content
            combined_content = (
                f"<think>\n{reasoning_content}\n</think>\n"
                if reasoning_content
                else ""
            ) + response.choices[0].message.content

            response = ChatCompletion.construct(
                id=response.id,
                choices=[
                    dict(
                        index=response.choices[0].index,
                        message={
                            "role": response.choices[0].message.role,
                            "content": combined_content,
                            "tool_calls": None,
                        },
                        finish_reason=response.choices[0].finish_reason
                        if response.choices[0].finish_reason
                        else None,
                    )
                ],
                created=response.created,
                model=response.model,
                object="chat.completion",
                usage=response.usage,
            )

        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to DeepSeek API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to DeepSeek API.
        """
        for param in self.model_config_dict:
            if param not in DEEPSEEK_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into DeepSeek model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get("stream", False)

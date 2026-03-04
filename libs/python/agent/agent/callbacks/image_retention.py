"""
Image retention callback handler that limits the number of recent images in message history.
"""

from typing import Any, Dict, List, Optional

from .base import AsyncCallbackHandler


class ImageRetentionCallback(AsyncCallbackHandler):
    """
    Callback handler that applies image retention policy to limit the number
    of recent images in message history to prevent context window overflow.
    """

    def __init__(self, only_n_most_recent_images: Optional[int] = None):
        """
        Initialize the image retention callback.

        Args:
            only_n_most_recent_images: If set, only keep the N most recent images in message history
        """
        self.only_n_most_recent_images = only_n_most_recent_images

    async def on_llm_start(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply image retention policy to messages before sending to agent loop.

        Args:
            messages: List of message dictionaries

        Returns:
            List of messages with image retention policy applied
        """
        if self.only_n_most_recent_images is None:
            return messages

        return self._apply_image_retention(messages)

    def _apply_image_retention(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply image retention policy to keep only the N most recent images.

        For older images beyond the retention window, replaces the image_url with
        "[omitted]" instead of removing the entire action triplet. This preserves
        the model's memory of what actions were taken (reasoning + computer_call)
        while reducing token usage from old screenshots.

        Args:
            messages: List of message dictionaries

        Returns:
            List of messages with old image data replaced by "[omitted]" placeholders
        """
        if self.only_n_most_recent_images is None:
            return messages

        # Gather indices of all computer_call_output messages that contain an image_url
        output_indices: List[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("type") == "computer_call_output":
                out = msg.get("output")
                if isinstance(out, dict) and "image_url" in out and out["image_url"] != "[omitted]":
                    output_indices.append(idx)

        # Nothing to trim
        if len(output_indices) <= self.only_n_most_recent_images:
            return messages

        # Determine which outputs to keep (most recent N)
        keep_output_indices = set(output_indices[-self.only_n_most_recent_images :])

        # Replace old image data with "[omitted]" placeholder instead of removing
        # the entire action triplet. This preserves the model's procedural memory
        # (what actions were taken and why) while saving context window tokens.
        result = []
        for idx, msg in enumerate(messages):
            if idx in output_indices and idx not in keep_output_indices:
                # Shallow copy the message to avoid mutating the original
                omitted_msg = {
                    **msg,
                    "output": {**msg["output"], "image_url": "[omitted]"},
                }
                result.append(omitted_msg)
            else:
                result.append(msg)

        return result

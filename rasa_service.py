import os
import aiohttp
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

RASA_URL = os.getenv("RASA_URL", "http://0.0.0.0:5055")

async def analyze_text_with_rasa(text: str) -> Dict[str, Any]:
    """
    Phân tích văn bản sử dụng Rasa
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{RASA_URL}/webhooks/rest/webhook",
                json={"message": text}
            ) as response:
                if response.status == 200:
                    rasa_response = await response.json()
                    return {
                        "status": "success",
                        "rasa_response": rasa_response
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Failed to analyze text with Rasa"
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error communicating with Rasa: {str(e)}"
            }

def extract_device_command(rasa_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trích xuất lệnh điều khiển thiết bị từ phản hồi của Rasa
    """
    if not rasa_response.get("rasa_response"):
        return {
            "action": None,
            "device_name": None
        }

    for response in rasa_response["rasa_response"]:
        if "custom" in response:
            return {
                "action": response["custom"].get("action"),
                "device_name": response["custom"].get("device_name"),
                "value": response["custom"].get("value")
            }
    
    return {
        "action": None,
        "device_name": None
    } 
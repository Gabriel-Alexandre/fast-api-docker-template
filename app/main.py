from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List
import requests
import base64

load_dotenv()

app = FastAPI()

class StoryPrompt(BaseModel):
    prompt: str
    num_images: int = 4

class StoryResponse(BaseModel):
    images: List[str]
    prompt: str

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story(story_prompt: StoryPrompt):
    try:
        api_host = 'https://api.stability.ai'
        api_key = os.getenv('STABILITY_KEY')
        
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not found. Please check your .env file")

        print(f"Fazendo requisição para Stability AI com prompt: {story_prompt.prompt}")
        
        response = requests.post(
            f"{api_host}/v1/generation/stable-diffusion-v1-6/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "text_prompts": [{"text": story_prompt.prompt}],
                "samples": story_prompt.num_images,
                "steps": 20,
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
            },
        )

        print(f"Status code da resposta: {response.status_code}")
        print(f"Resposta da API: {response.text}")

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Stability AI API error: {response.text}"
            )

        data = response.json()
        generated_images = []

        os.makedirs("generated_images", exist_ok=True)
        
        for idx, image in enumerate(data["artifacts"]):
            img_path = f"generated_images/image_{idx}.png"
            img_data = base64.b64decode(image["base64"])
            
            with open(img_path, "wb") as f:
                f.write(img_data)
            
            generated_images.append(img_path)

        return StoryResponse(
            images=generated_images,
            prompt=story_prompt.prompt
        )

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}") 
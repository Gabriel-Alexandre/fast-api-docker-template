from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List
import requests
import base64
import json

load_dotenv()

app = FastAPI()

class StoryPrompt(BaseModel):
    prompt: str

class StoryScene(BaseModel):
    description: str
    image_path: str = None

class StoryResponse(BaseModel):
    title: str
    scenes: List[StoryScene]
    original_prompt: str

def generate_story_with_ollama(prompt: str) -> List[dict]:
    """Gera uma história com 8 cenas usando Ollama"""
    system_prompt = """Você é um escritor criativo. Crie uma história baseada no prompt fornecido.
    A história deve ser dividida em exatamente 8 cenas. Para cada cena, forneça uma descrição detalhada
    que possa ser usada para gerar uma imagem. Retorne no seguinte formato JSON:
    {
        "title": "Título da História",
        "scenes": [
            {"description": "Descrição detalhada da cena 1"},
            {"description": "Descrição detalhada da cena 2"},
            // ... até 8 cenas
        ]
    }"""

    try:
        response = requests.post('http://localhost:11434/api/generate',
            json={
                "model": "llama2",
                "prompt": f"{system_prompt}\n\nPrompt: {prompt}\n\nGere a história em formato JSON:",
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Erro ao gerar história com Ollama")

        # Extrair o JSON da resposta do Ollama
        response_text = response.json()['response']
        # Encontrar o JSON na resposta
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        return json.loads(json_str)

    except Exception as e:
        print(f"Erro ao gerar história: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar história: {str(e)}")

async def generate_image_with_stability(prompt: str) -> str:
    """Gera uma imagem usando Stability AI"""
    api_host = 'https://api.stability.ai'
    api_key = os.getenv('STABILITY_KEY')

    if not api_key:
        raise HTTPException(status_code=500, detail="API key not found")

    try:
        response = requests.post(
            f"{api_host}/v1/generation/stable-diffusion-v1-6/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "samples": 1,
                "steps": 20,
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                              detail=f"Stability AI API error: {response.text}")

        data = response.json()
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        
        # Criar diretório se não existir
        os.makedirs("generated_images", exist_ok=True)
        
        # Gerar nome único para a imagem
        image_path = f"generated_images/image_{len(os.listdir('generated_images'))}.png"
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        return image_path

    except Exception as e:
        print(f"Erro ao gerar imagem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar imagem: {str(e)}")

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story(story_prompt: StoryPrompt):
    try:
        # Gerar história com Ollama
        story_data = generate_story_with_ollama(story_prompt.prompt)
        
        # Lista para armazenar as cenas com suas imagens
        scenes = []
        
        # Gerar imagem para cada cena
        for scene in story_data["scenes"]:
            # Gerar imagem baseada na descrição da cena
            image_path = await generate_image_with_stability(scene["description"])
            
            # Adicionar cena com imagem à lista
            scenes.append(StoryScene(
                description=scene["description"],
                image_path=image_path
            ))

        return StoryResponse(
            title=story_data["title"],
            scenes=scenes,
            original_prompt=story_prompt.prompt
        )

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}") 
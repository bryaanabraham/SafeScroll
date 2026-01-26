from ollama import chat

def get_time_from_image(image_paths: list[str], model: str = 'gemma3:4b'):
    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What time does it look like it is in this image? Answer in ranges: like '1PM-2PM' or '2PM-3PM' etc.",
                    "images": image_paths
                }
            ]
        )
        print(f"Time output: {response.message.content}")
        print("="*50)
        return response.message.content
    except Exception as e:
        print(f"Error occured: {e}")
        return "Error occured while getting the time of the day, proceed without it."
    
def summarize_image(image_paths: list[str], model: str = 'gemma3:4b'):
    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the contents of this image",
                    "images": image_paths
                }
            ]
        )
        print(f"Summary output: {response.message.content}")
        print("="*50)
        return response.message.content
    except Exception as e:
        print(f"Error occured: {e}")
        return "Error occured while getting the image summary, proceed without it."

def context_aware_ocr(image_paths: list[str], model: str = 'gemma3:4b'):
    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Act as a context-aware OCR model and give me all the text you see in this image",
                    "images": image_paths
                }
            ]
        )

        print(f"OCR output: {response.message.content}")
        print("="*50)
        return response.message.content
    except Exception as e:
        print(f"Error occured: {e}")
        return "Error occured while performing OCR, proceed without it."


def complete_summary(time_content:str, summary_content:str, text_content:str, caption_content:str, face_data:str, model:str="llama3.1:8b"):
    
    response = chat(
        model=model,
        messages=[{'role': 'user', 'content': f'''
Based on the information provided about the time, image contents, text on the image, image captions and face data summarize all the information and produce a collated reported about what information can be infered from this:
Time: {time_content}
Summary: {summary_content}
Text on Image: {text_content}
Image Caption: {caption_content}
Faces: {face_data}                
            '''}],
    )
    print(f"OCR output: {response.message.content}")
    print("="*50)
    return response.message.content
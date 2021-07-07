from fastapi import FastAPI
from language_identification.language_identification import LanguageIdentification
import uvicorn

app = FastAPI()

language_identification_ = LanguageIdentification()


@app.get("/language_identification")
def language_identification(text):
    return language_identification_.predict(text)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

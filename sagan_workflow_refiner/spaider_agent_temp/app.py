from langchain_core.runnables.config import RunnableConfig
# LOCAL IMPORTS.
from graph import create_graph, compile_graph, print_stream
from schemas import State
from prompts.prompts import *

# api handling imports.
# from fastapi import FastAPI,File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from pathlib import Path
# from pypdf import PdfReader


# app = FastAPI()

# DATA_FOLDER = Path("testfolder")
# DATA_FOLDER.mkdir(exist_ok=True)

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     if file.content_type != "application/pdf":
#         return JSONResponse(status_code=400, content={"message": "Invalid file type. Please upload a PDF."})

#     file_location = DATA_FOLDER / file.filename
#     with open(file_location, "wb") as f:
#         f.write(await file.read())

#     # Validate the PDF and read all pages
#     try:
#         with open(file_location, "rb") as pdf_file:
#             reader = PdfReader(pdf_file)
#             # Iterate through all pages to ensure the PDF is fully readable
#             for page_num, page in enumerate(reader.pages):
#                 text = page.extract_text()
#                 print(f"Text from page {page_num + 1}: {text}")
#     except Exception as e:
#         # Delete the file if it's not a valid PDF
#         os.remove(file_location)
#         return JSONResponse(status_code=400, content={"message": "The file is not a valid PDF."})

#     return JSONResponse(content={"message": "File uploaded and verified successfully!"})





config = RunnableConfig(
    recursion_limit=50,
    configurable={"thread_id": "1"}
)
print(config)



if __name__ == "__main__":
    # creating graph workflow instance and then compiling it.
    verbose = True
    builder = create_graph()
    graph = compile_graph(builder)

    # Create a State object with the given section_title and section_text
    section_title="Context and Motivation"
    section_text="The University of Luxembourg, through its Research Unit in Engineering Science (RUES), is committed to addressing the socio-economic needs and challenges of society and industry by becoming a leader in education and research in the Greater Region and globally. The unit focuses on three main research areas: Construction and Design, Energy and Environment, and Automation and Mechatronics. These areas encompass research into civil and mechanical engineering structures, energy efficiency, renewable energies, and dynamic testing methods, among others. The university aims to seamlessly integrate research and education to cultivate future leaders and critical thinkers. In collaboration with over 70 private and public organizations through SnT's Partnership Programme, the university addresses key challenges in ICT, contributing to the European Strategic Technology Plan and the Innovation Union in Europe. Since its launch in 2009, the Centre has rapidly developed, launching over 100 EU and ESA projects, protecting and licensing IP, and creating a dynamic interdisciplinary research environment with around 480 people. For all AFR individual applications, a project idea must be outlined using a specific template, detailing the hypothesis, research questions, innovation, expected outcomes, and methodology. The FNR encourages the dissemination of research to the public and media, emphasizing the value and impact of research outputs. This approach ensures that research activities are aligned with industry, policymakers, and societal needs, fostering an innovation-driven research environment."  

    researcher_prompt = RESEARCHER_PROMPT.format(section_title=section_title, section_text=section_text)

    # print the mermaid diagram of the graph.
    print(graph.get_graph().draw_mermaid())

    # taking user input.
    user_input = input("############# User: ")

    # initting state with the formatted researcher prompt, and section title and text.
    initial_state = {"messages": [("system", researcher_prompt), ("user", user_input)], "section_title": section_title, "section_text": section_text}

    # Pass the State object to the graph.stream method
    print_stream(graph.stream(initial_state, stream_mode="values", config=config))

import traceback

import click
import os

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import io

from fastapi import FastAPI, Form, File, UploadFile, Depends
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ] = None
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        Optional[bool],
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        Optional[bool],
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        Optional[str],
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"
    output_dir: Annotated[
        Optional[str],
        Field(
            description="Directory where output files will be saved. Defaults to the value specified in settings.OUTPUT_DIR."
        ),
    ] = None
    use_llm: Annotated[
        Optional[bool],
        Field(
            description="Uses an LLM to improve accuracy. You will need to configure the LLM backend."
        ),
    ] = False
    format_lines: Annotated[
        Optional[bool],
        Field(
            description="Reformat all lines using a local OCR model (inline math, underlines, bold, etc.). This will give very good quality math output."
        ),
    ] = False
    block_correction_prompt: Annotated[
        Optional[str],
        Field(
            description="if LLM mode is active, an optional prompt that will be used to correct the output of marker. This is useful for custom formatting or logic that you want to apply to the output."
        ),
    ] = None
    strip_existing_ocr: Annotated[
        Optional[bool],
        "Remove all existing OCR text in the document and re-OCR with surya.",
    ] = False
    redo_inline_math: Annotated[
        Optional[bool],
        "If you want the absolute highest quality inline math conversion, use this along with --use_llm.",
    ] = False
    disable_image_extraction: Annotated[
        Optional[bool],
        "Don't extract images from the PDF. If you also specify --use_llm, then images will be replaced with a description.",
    ] = False
    debug: Annotated[
        Optional[bool],
        "Enable debug mode for additional logging and diagnostic information.",
    ] = False
    processors: Annotated[
        Optional[str],
        Field(
            description="Override the default processors by providing their full module paths, separated by commas. Example: --processors 'module1.processor1,module2.processor2'."
        ),
    ] = None
    config_json: Annotated[
        Optional[str],
        Field(
            description="Path to a JSON configuration file containing additional settings."
        ),
    ] = None
    converter_cls: Annotated[
        Optional[str],
        Field(
            description="One of marker.converters.pdf.PdfConverter (default) or marker.converters.table.TableConverter. The PdfConverter will convert the whole PDF, the TableConverter will only extract and convert tables."
        ),
    ] = None
    llm_service: Annotated[
        Optional[str],
        Field(
            description="Which llm service to use if --use_llm is passed. This defaults to marker.services.gemini.GoogleGeminiService."
        ),
    ] = None
    gemini_api_key: Annotated[
        Optional[str],
        Field(
            description="this will use the Gemini developer API by default."
        ),
    ] = None
    vertex_project_id: Annotated[
        Optional[str],
        Field(
            description="this will use vertex, which can be more reliable, To use it, set --llm_service=marker.services.vertex.GoogleVertexService."
        ),
    ] = None
    ollama_base_url: Annotated[
        Optional[str],
        Field(
            description="this will use local models, used with ollama_model, To use it, set --llm_service=marker.services.ollama.OllamaService."
        ),
    ] = None
    ollama_model: Annotated[
        Optional[str],
        Field(
            description="this will use local models, used with ollama_base_url, To use it, set --llm_service=marker.services.ollama.OllamaService."
        ),
    ] = None
    claude_api_key: Annotated[
        Optional[str],
        Field(
            description="this will use the anthropic API. You can configure --claude_api_key, and --claude_model_name. To use it, set --llm_service=marker.services.claude.ClaudeService."
        ),
    ] = None
    claude_model_name: Annotated[
        Optional[str],
        Field(
            description="this will use the anthropic API. You can configure --claude_api_key, and --claude_model_name. To use it, set --llm_service=marker.services.claude.ClaudeService."
        ),
    ] = None
    openai_api_key: Annotated[
        Optional[str],
        Field(
            description="this supports any openai-like endpoint. You can configure --openai_api_key, --openai_model, and --openai_base_url. To use it, set --llm_service=marker.services.openai.OpenAIService."
        ),
    ] = None
    openai_model: Annotated[
        Optional[str],
        Field(
            description="this supports any openai-like endpoint. You can configure --openai_api_key, --openai_model, and --openai_base_url. To use it, set --llm_service=marker.services.openai.OpenAIService."
        ),
    ] = None
    openai_base_url: Annotated[
        Optional[str],
        Field(
            description="this supports any openai-like endpoint. You can configure --openai_api_key, --openai_model, and --openai_base_url. To use it, set --llm_service=marker.services.openai.OpenAIService."
        ),
    ] = None
    azure_endpoint: Annotated[
        Optional[str],
        Field(
            description="this uses the Azure OpenAI service. You can configure --azure_endpoint, --azure_api_key, and --deployment_name. To use it, set --llm_service=marker.services.azure_openai.AzureOpenAIService."
        ),
    ] = None
    azure_api_key: Annotated[
        Optional[str],
        Field(
            description="this uses the Azure OpenAI service. You can configure --azure_endpoint, --azure_api_key, and --deployment_name. To use it, set --llm_service=marker.services.azure_openai.AzureOpenAIService."
        ),
    ] = None
    deployment_name: Annotated[
        Optional[str],
        Field(
            description="this uses the Azure OpenAI service. You can configure --azure_endpoint, --azure_api_key, and --deployment_name. To use it, set --llm_service=marker.services.azure_openai.AzureOpenAIService."
        ),
    ] = None

    @classmethod
    def as_form(cls,
                page_range: Optional[str] = Form(default=None),
                force_ocr: Optional[bool] = Form(default=False),
                paginate_output: Optional[bool] = Form(default=False),
                output_format: Optional[str] = Form(default="markdown"),
                output_dir: Optional[str] = Form(default=None),
                use_llm: Optional[bool] = Form(default=False),
                format_lines: Optional[bool] = Form(default=False),
                block_correction_prompt: Optional[str] = Form(default=None),
                strip_existing_ocr: Optional[bool] = Form(default=False),
                redo_inline_math: Optional[bool] = Form(default=False),
                disable_image_extraction: Optional[bool] = Form(default=False),
                debug: Optional[bool] = Form(default=False),
                processors: Optional[str] = Form(default=None),
                config_json: Optional[str] = Form(default=None),
                converter_cls: Optional[str] = Form(default=None),
                llm_service: Optional[str] = Form(default=None),
                gemini_api_key: Optional[str] = Form(default=None),
                vertex_project_id: Optional[str] = Form(default=None),
                ollama_base_url: Optional[str] = Form(default=None),
                ollama_model: Optional[str] = Form(default=None),
                claude_api_key: Optional[str] = Form(default=None),
                claude_model_name: Optional[str] = Form(default=None),
                openai_api_key: Optional[str] = Form(default=None),
                openai_model: Optional[str] = Form(default=None),
                openai_base_url: Optional[str] = Form(default=None),
                azure_endpoint: Optional[str] = Form(default=None),
                azure_api_key: Optional[str] = Form(default=None),
                deployment_name: Optional[str] = Form(default=None)):
        return cls(
            page_range=page_range,
            force_ocr=force_ocr,
            paginate_output=paginate_output,
            output_format=output_format,
            output_dir=output_dir,
            use_llm=use_llm,
            format_lines=format_lines,
            block_correction_prompt=block_correction_prompt,
            strip_existing_ocr=strip_existing_ocr,
            redo_inline_math=redo_inline_math,
            disable_image_extraction=disable_image_extraction,
            debug=debug,
            processors=processors,
            config_json=config_json,
            converter_cls=converter_cls,
            llm_service=llm_service,
            gemini_api_key=gemini_api_key,
            vertex_project_id=vertex_project_id,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            claude_api_key=claude_api_key,
            claude_model_name=claude_model_name,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            deployment_name=deployment_name
        )


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html", "chunks"], (
        "Invalid output format"
    )
    try:
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        # The number of workers to use for pdftext default is 4
        config_dict["pdftext_workers"] = int(os.getenv('WORKER_NUM', '4'))
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
        params: CommonParams = Depends(CommonParams.as_form),
        file: UploadFile = File(
            ..., description="The PDF file to convert.", media_type="application/pdf"
        ),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params.filepath = upload_path
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )

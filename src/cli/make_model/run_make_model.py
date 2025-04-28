import click

from ...warboy.tools.onnx_tools import OnnxTools


@click.command(
    "make-model",
    help="Make quantized model from config file.",
    short_help="Make quantized ONNX model",
)
@click.argument("config_file")
@click.option(
    "--need_edit",
    is_flag=True,
    default=True,
    help="Whether you need to edit the model.",
)
@click.option(
    "--need_quantize",
    is_flag=True,
    default=True,
    help="Whether you need to quantize the model.",
)
def run_make_model(config_file: str, need_edit: bool, need_quantize: bool):
    onnx_tools = OnnxTools(config_file)

    if need_edit and "yolo" not in onnx_tools.model_name:
        click.echo(
            "Warning: The model is not a YOLO model. The need_edit option is ignored."
        )
        need_edit = False
    onnx_tools.export_onnx(need_edit=need_edit)

    if need_quantize:
        onnx_tools.quantize()


@click.command(
    "export-onnx", help="Export ONNX model from config file.", short_help="Export ONNX"
)
@click.argument("config_file")
@click.option(
    "--need_edit",
    is_flag=True,
    default=True,
    help="Whether you need to edit the model.",
)
def run_export_onnx(config_file: str, need_edit: bool):
    onnx_tools = OnnxTools(config_file)

    if need_edit and "yolo" not in onnx_tools.model_name:
        click.echo(
            "Warning: The model is not a YOLO model. The need_edit option is ignored."
        )
        need_edit = False

    onnx_tools.export_onnx(need_edit=need_edit)


@click.command(
    "quantize",
    help="Quantize ONNX model from config file. To run this command, you need to prepare the ONNX model first.",
    short_help="Quantize ONNX, need ONNX model first",
)
@click.argument("config_file")
def run_quantize(config_file: str):
    onnx_tools = OnnxTools(config_file)
    onnx_tools.quantize()

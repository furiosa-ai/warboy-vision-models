import click

from ...warboy.tools.onnx_tools import OnnxTools


@click.command(
    "make-model",
    help="Make quantized model from config file.",
    short_help="Make quantized ONNX model",
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="The path to the model configuration file.",
)
@click.option(
    "--need_edit/--no-need-edit",
    default=True,
    help="Editing is enabled by default. Use '--no-need-edit' to disable it.",
)
@click.option(
    "--need_quantize/--no-need-quantize",
    default=True,
    help="Quantization is enabled by default. Use '--no-need-quantize' to disable it.",
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
@click.option(
    "--config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="The path to the model configuration file.",
)
@click.option(
    "--need_edit/--no-need-edit",
    default=True,
    help="Editing is enabled by default. Use '--no-need-edit' to disable it.",
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
@click.option(
    "--config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="The path to the model configuration file.",
)
def run_quantize(config_file: str):
    onnx_tools = OnnxTools(config_file)
    onnx_tools.quantize()

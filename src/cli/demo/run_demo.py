import click

from ...demo import demo


@click.command(
    "run-demo",
    help="Run the demo by web or making output files.",
    short_help="Run demo, mode is web or file.",
)
@click.option(
    "--demo_config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="The path to the demo configuration file.",
)
@click.option(
    "--mode",
    type=click.Choice(["web", "image", "file"], case_sensitive=False),
    default="web",
    help="Choose the mode to run the demo: 'web' or 'file' or 'image'. Default is 'web'.",
)
def run_demo(demo_config_file: str, mode: str):
    if mode == "web":
        demo.run_web_demo(demo_config_file)

    if mode == "image":
        demo.run_make_image(demo_config_file)

    elif mode == "file":
        demo.run_make_file(demo_config_file)

    else:
        raise ValueError("Invalid mode. Choose 'web' or 'file'.")

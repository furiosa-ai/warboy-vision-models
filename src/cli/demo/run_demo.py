import click

from ...demo import demo


@click.command("run-demo")
@click.argument("demo_config_file")
@click.option(
    "--mode",
    type=click.Choice(["web", "file"], case_sensitive=False),
    default="web",
    help="Choose the mode to run the demo: 'web' or 'file'.",
)
def run_demo(demo_config_file: str, mode: str):
    if mode == "web":
        demo.run_web_demo(demo_config_file)

    elif mode == "file":
        demo.run_make_file(demo_config_file)

    else:
        raise ValueError("Invalid mode. Choose 'web' or 'file'.")

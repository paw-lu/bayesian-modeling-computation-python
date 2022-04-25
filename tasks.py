"""Tasks for aiding in notetaking."""
import pathlib

import invoke


@invoke.task(
    help={
        "chapter": "The chapture number."
        " Determines the directory the screenshot is saved under.",
        "name": "The new name of the file. Automatically adds a '.png' extension.",
    }
)
def screenshot(context, chapter, name=None):
    """Move screenshot to the images directory and copy relative path."""
    screenshots = list((pathlib.Path.home() / pathlib.Path("Desktop")).glob("*.png"))
    if (number_of_screenshots := len(screenshots)) < 1:
        raise ValueError("No screenshots found")
    elif 1 < number_of_screenshots:
        raise ValueError("Multiple screenshots found")

    screenshot_path, *_ = screenshots
    project_path = pathlib.Path(__file__).parent / pathlib.Path("notebooks")
    new_file_name = (
        pathlib.Path(screenshot_path.name)
        if name is None
        else pathlib.Path(name).with_suffix(".png")
    )
    image_dir_name = pathlib.Path("images", f"chapter_{chapter}")
    (project_path / image_dir_name ).mkdir(parents=True, exist_ok=True)
    relative_new_path = image_dir_name / new_file_name
    absolute_new_path = project_path / relative_new_path
    context.run(f"mv {screenshot_path} {absolute_new_path}")
    context.run(f"echo '![]({relative_new_path})' | pbcopy")

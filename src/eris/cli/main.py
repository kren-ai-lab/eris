from __future__ import annotations

import typer

from eris.cli.cache import app as cache_app
from eris.cli.encode import app as encode_app
#from eris.cli.pack import app as pack_app
from eris.cli.info_cmd import app as info_app
from eris.cli.encoders import app as encoders_app
from eris.cli.export import app as torch_app

app = typer.Typer(add_completion=False)

# register subcommands
app.add_typer(cache_app, name="cache")
app.add_typer(encode_app, name="encode")
#app.add_typer(pack_app, name="pack")
app.add_typer(info_app, name="info")
app.add_typer(encoders_app, name="encoders")
app.add_typer(torch_app, name="export")

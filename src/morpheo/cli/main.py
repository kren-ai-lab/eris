from __future__ import annotations

import typer

from morpheo.cli.cache import app as cache_app
from morpheo.cli.encode import app as encode_app
from morpheo.cli.batch_cmd import app as batch_app
#from morpheo.cli.pack import app as pack_app
from morpheo.cli.info_cmd import app as info_app
from morpheo.cli.encoders import app as encoders_app

app = typer.Typer(add_completion=False)

# register subcommands
app.add_typer(cache_app, name="cache")
app.add_typer(encode_app, name="encode")
app.add_typer(batch_app, name="batch")
#app.add_typer(pack_app, name="pack")
app.add_typer(info_app, name="info")
app.add_typer(encoders_app, name="encoders")

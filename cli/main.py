import click
from .games.pong_arcade import pong
from .games.space_invaders_arcade import space_invaders

@click.group()
def cli():
    """Agent Arcade CLI"""
    pass

cli.add_command(pong)
cli.add_command(space_invaders)

if __name__ == '__main__':
    cli() 
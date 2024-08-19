"""
This is a discord bot. It should read progress
from a file given in FILENAME and send it to a
discord channel.
"""

import asyncio
import json
import os
from typing import Dict, Optional, Tuple

import discord
from discord.ext import commands

bot = commands.Bot(command_prefix="/",
                   intents=discord.Intents.all())

SPRL_PATH = "/home/gridsan/rzhong/sprl"

global_progress = {}


async def parse_all_running():
    """
    Parse all running jobs in the SPRL_PATH.
    """

    # iterate over all files in the logs directory
    # if they are of the form "name_<index>.log", then parse them.
    new_progress = {}

    for file in os.listdir(os.path.join(SPRL_PATH, 'logs')):
        if not file.endswith(".log"):
            continue
        name = file.rsplit(".", 1)[0]
        name = name.rsplit("_", 1)[0]

        with open(os.path.join(SPRL_PATH, 'logs', file)) as f:
            lines = f.readlines()
        iteration = 0
        for line in lines:
            if "Starting iteration" in line:
                iteration = int(line.split()[-1][:-3])  # remove the ellipsis

            if "finished training." in line:
                # A sort of hacky way to do this.
                iteration += 1
        try:
            with open(f"{SPRL_PATH}/data/configs/{name}_config_selfplay.json") as f:
                config = json.load(f)
                total_iters = config["numIters"]
                new_progress[name] = [iteration, total_iters]
        except FileNotFoundError:
            new_progress[name] = [iteration, None]

    return new_progress


# async def get_all_tournaments():
#     """
#     Check the robin folder and look for all
#     """


def progress_to_embed(progress: Dict[str, Tuple[int, Optional[int]]], color=discord.Color.blue()):
    """
    Convert the progress to a discord embed.
    """
    embed = discord.Embed(title="Progress", color=color)
    for name, iteration in progress.items():
        if iteration[0] == 0:
            embed.add_field(name=name, value="Initializing... :arrow_forward:")
        if iteration[1] is not None:
            if iteration[0] >= iteration[1]:
                embed.add_field(name=name, value="Done! :trophy:")
            else:
                embed.add_field(name=name,
                                value=f"Iteration {iteration[0]}/{iteration[1]}")
        else:
            embed.add_field(name=name, value=f"Iteration {iteration[0]}")
    return embed


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    bot.loop.create_task(watch_file())


@bot.command()
async def progress(ctx: commands.Context):
    """
    Send the progress of all running jobs.
    """
    # print the channel from which the command was sent
    print(ctx.channel.id)

    progress = await parse_all_running()
    # Send the progress in an embed
    await ctx.send(embed=progress_to_embed(progress))


# This command should be triggered by either
# !json or !info
@bot.command(aliases=["json"])
async def info(ctx: commands.Context, name: str):
    """
    Send the infos of all running jobs.

    # Next, for each running job, go to data/configs
    # and find three files:
    #  /data/configs/{name}_config_selfplay.json
    #  /data/configs/{name}_config_controller.json
    #  /data/configs/{name}_config_uct.json
    # This is a json file; you should parse it.
    # Output each json file in a pretty embed.
    """

    # make sure name is alphanumeric with underscores.
    flag = False
    for c in name:
        if not c.isalnum() and c != "_":
            flag = True
            break
    if flag:
        await ctx.send("Invalid name. Please use only alphanumeric characters and underscores.")
        return
    config_path = f"{SPRL_PATH}/data/configs"
    selfplay_filepath = f"{config_path}/{name}_config_selfplay.json"
    controller_filepath = f"{config_path}/{name}_config_controller.json"
    uct_filepath = f"{config_path}/{name}_config_uct.json"

    with open(selfplay_filepath) as f:
        selfplay: Dict = json.load(f)

    with open(controller_filepath) as f:
        controller = json.load(f)

    with open(uct_filepath) as f:
        uct = json.load(f)

    # Send 3 separate embeds
    selfplay_embed = discord.Embed(title=f"{name}_config_selfplay.json")
    for key in selfplay.keys():
        selfplay_embed.add_field(name=key, value=selfplay[key])

    controller_embed = discord.Embed(title=f"{name}_config_controller.json")
    for key in controller.keys():
        controller_embed.add_field(name=key, value=controller[key])

    uct_embed = discord.Embed(title=f"{name}_config_uct.json")
    for key in uct.keys():
        uct_embed.add_field(name=key, value=uct[key])

    await ctx.send(embed=selfplay_embed)
    await ctx.send(embed=controller_embed)
    await ctx.send(embed=uct_embed)


async def watch_file():
    """
    Watch the progress file and send it to the channel.
    """
    global global_progress
    channel = bot.get_channel(1259883219046764615)
    # int(os.getenv("CHANNEL_ID")))
    while True:
        # Check the progress. If it is different from before, send it to the channel.
        # Edge case: if a process existed before but does not anymore, we do not send anything!
        progress = await parse_all_running()
        completed_tasks = []
        changed = False
        for name, iteration in progress.items():
            if name not in global_progress or global_progress[name][0] != iteration[0]:
                changed = True
                if iteration[1] is not None and iteration[0] >= iteration[1]:
                    completed_tasks.append(name)
        global_progress = progress
        if changed:
            await channel.send(embed=progress_to_embed(progress))

        # Sleep for 5 minutes (usually, changed is False)
        await asyncio.sleep(300)


# read discord token from a json file
with open(f"{SPRL_PATH}/dopamine/config.json") as f:
    data = json.load(f)
    DISCORD_TOKEN = data["DISCORD_TOKEN"]
bot.run(DISCORD_TOKEN)

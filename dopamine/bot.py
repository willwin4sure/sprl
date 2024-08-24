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
from discord import app_commands


SPRL_PATH = os.path.dirname(os.path.realpath(__file__))
SPRL_PATH = os.path.dirname(SPRL_PATH)
print(f"SPRL_PATH: {SPRL_PATH}")

# Read Discord token from gitignored config file.
with open(f"{SPRL_PATH}/dopamine/config.json") as f:
    data = json.load(f)
    DISCORD_TOKEN = data["DISCORD_TOKEN"]  # Sensitive data.
    GUILD_ID = data["GUILD_ID"]
    CHANNEL_ID = data["CHANNEL_ID"]


intents = discord.Intents.all()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


global_progress = {}

async def parse_all_running():
    """Parse all running jobs in the SPRL_PATH."""
    
    progress = {}

    # Parse files of the form "name_<index>.log" in the `logs` directory.
    for file in os.listdir(os.path.join(SPRL_PATH, "logs")):
        if not file.endswith(".log"):
            continue
        
        name = file.rsplit(".", 1)[0]
        name = name.rsplit("_", 1)[0]

        with open(os.path.join(SPRL_PATH, "logs", file)) as f:
            lines = f.readlines()

        iteration = 0
        for line in lines:
            if "Starting iteration" in line:
                iteration = int(line.split()[-1][:-3])  # Remove the ellipsis.

            if "finished training." in line:
                iteration += 1  # If the bot finished training.

        try:
            with open(f"{SPRL_PATH}/data/configs/{name}_config_selfplay.json") as f:
                config = json.load(f)
                total_iters = config["numIters"]
                progress[name] = [iteration, total_iters]

        except FileNotFoundError:
            progress[name] = [iteration, None]

    return progress


def progress_to_embed(progress: Dict[str, Tuple[int, Optional[int]]], color=discord.Color.blue()):
    """Convert the progress to a discord embed."""

    embed = discord.Embed(title="Progress", color=color)
    for name, iteration in progress.items():
        if iteration[0] == 0:
            embed.add_field(name=name, value="Initializing... :arrow_forward:")

        if iteration[1] is not None:
            if iteration[0] >= iteration[1]:
                embed.add_field(name=name, value="Done! :trophy:")
            else:
                embed.add_field(name=name, value=f"Iteration {iteration[0]}/{iteration[1]}")

        else:
            embed.add_field(name=name, value=f"Iteration {iteration[0]}")

    return embed


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}.")

    # Only sync the tree if commands have changed.
    await tree.sync(guild=discord.Object(id=GUILD_ID))

    # Start watching the progress file.
    bot.loop.create_task(watch_file())


@tree.command(
    name="progress",
    description="Immediately send the progress of all running jobs.",
    guild=discord.Object(id=GUILD_ID),
)
async def progress(interaction: discord.Interaction):
    """Immediately send the progress of all running jobs."""

    await interaction.response.defer()

    # Send the progress in an embed.
    progress = await parse_all_running()
    embed = progress_to_embed(progress)

    await interaction.followup.send(embed=embed)


async def get_job_name_autocomplete(interaction: discord.Interaction, current: str):
    """Get the job names for the autocomplete."""

    job_names = []
    for file in os.listdir(os.path.join(SPRL_PATH, "data", "configs")):
        if not file.endswith(".json"):
            continue

        name = file.rsplit("_config_", 1)[0]
        if current in name and name not in job_names:
            job_names.append(name)

    return [
        app_commands.Choice(name=name, value=name)
        for name in job_names
    ]


@tree.command(
    name="info",
    description="Dumps the config files of a job.",
    guild=discord.Object(id=GUILD_ID),
)
@app_commands.describe(
    job_name="The name of the job to get the config files of."
)
@app_commands.autocomplete(
    job_name=get_job_name_autocomplete
)
async def info(interaction: discord.Interaction, job_name: str):
    """Dumps the config files of a job.

    For each running job, goes to `data/configs` and dumps:
        /data/configs/{name}_config_selfplay.json
        /data/configs/{name}_config_controller.json
        /data/configs/{name}_config_uct.json
    """

    await interaction.response.defer()

    # Check if the name is valid.
    invalid = False
    for c in job_name:
        if not c.isalnum() and c != "_":
            invalid = True
            break

    if invalid:
        await interaction.followup.send("Invalid name. Please use only alphanumeric characters and underscores.")
        return
    
    config_path = os.path.join(SPRL_PATH, "data", "configs")

    selfplay_filepath = os.path.join(config_path, f"{job_name}_config_selfplay.json")
    controller_filepath = os.path.join(config_path, f"{job_name}_config_controller.json")
    uct_filepath = os.path.join(config_path, f"{job_name}_config_uct.json")

    try:
        with open(selfplay_filepath) as f:
            selfplay: Dict = json.load(f)

        with open(controller_filepath) as f:
            controller: Dict = json.load(f)

        with open(uct_filepath) as f:
            uct: Dict = json.load(f)

    except FileNotFoundError:
        await interaction.followup.send("Config files not all found.")
        return

    # Send 3 separate embeds
    selfplay_embed = discord.Embed(title=f"{job_name}_config_selfplay.json")
    for key in selfplay.keys():
        selfplay_embed.add_field(name=key, value=selfplay[key])

    controller_embed = discord.Embed(title=f"{job_name}_config_controller.json")
    for key in controller.keys():
        controller_embed.add_field(name=key, value=controller[key])

    uct_embed = discord.Embed(title=f"{job_name}_config_uct.json")
    for key in uct.keys():
        uct_embed.add_field(name=key, value=uct[key])

    await interaction.followup.send(embeds=[selfplay_embed, controller_embed, uct_embed])


async def watch_file():
    """Watch the progress file and send it to the channel."""

    global global_progress
    channel = bot.get_channel(CHANNEL_ID)
    while True:
        print("Checking progress.")
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

        # Sleep for 5 minutes (usually, changed is False).
        await asyncio.sleep(300)


bot.run(DISCORD_TOKEN)
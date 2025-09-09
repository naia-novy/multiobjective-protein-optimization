
import itertools
import os
import shutil
import warnings
from os.path import join, isdir, isfile, basename
import time

import argparse

import yaml

from train import gen_model_uuid, log_dir_name
import utils


def create_run_dir(run_name="unnamed"):
    """ create run directory for this run """
    dir_name_str = "condor_{}_{}"
    run_dir_basename = dir_name_str.format(time.strftime("%Y-%m-%d_%H-%M-%S"), run_name)
    run_dir = join("output/htcondor_runs", run_dir_basename)
    os.makedirs(run_dir)
    print("Created run directory: {}".format(run_dir))
    return run_dir


def gen_args_from_template_fn(template_fn, out_dir):
    """ parse the given args template file and save individual args files in the out_dir """

    # if arguments already exist in the output directory, start the numbering to continue the args
    # this allows us to seamlessly parse multiple template files for a single run
    if isdir(out_dir):
        existing_fns = os.listdir(out_dir)
        max_arg_num = max([int(d[:-4]) for d in existing_fns]) + 1
    else:
        max_arg_num = 0
        os.makedirs(out_dir)

    args_dicts = parse_yml(template_fn)
    for i, d in enumerate(args_dicts):
        arg_num = max_arg_num + i
        utils.save_args(d, join(out_dir, "{}.txt".format(arg_num)))


def gen_args_from_template_fns(template_fns, args_out_dir):
    """ wrapper that allows multiple template args files for a single run """
    for template_fn in template_fns:
        gen_args_from_template_fn(template_fn, args_out_dir)

    num_jobs = len(os.listdir(args_out_dir))
    return num_jobs


def parse_yml(yml_fn):
    """ parses a yaml template args file into dictionaries """

    with open(yml_fn, "r") as f:
        yml = yaml.safe_load(f)

        # get the possible options for each arg, and make sure it's a list, even if it's just one possible option
        options = [val if isinstance(val, list) else [val] for val in yml.values()]
        combinations = list(itertools.product(*options))

        # create the dictionaries
        dicts = []
        for combo in combinations:
            d = dict(zip(yml.keys(), combo))
            dicts.append(d)

        # parse arguments that are supposed to be lists (i.e. strings delimited with commas)
        for d in dicts:
            for k, v in d.items():
                if isinstance(v, str) and "," in v:
                    d[k] = v.split(",")

        # assign UUIDs if any jobs have UUID of "condor"
        for d in dicts:
            if "uuid" in d and d["uuid"] == "condor":
                d["uuid"] = gen_model_uuid()
            elif "uuid" not in d:
                # generate model UUID if one was not specified...
                # should probably warn the user that we are doing this
                # in the future, will change system so a UUID is not needed to be in arg file
                # d["uuid"] = gen_model_uuid()
                pass

    return dicts


def save_env_vars(out_dir, github_tag, github_token, num_jobs, wandb_api_key):
    # create an env_vars.txt file to define environment variables
    # these can be set on the execute node via the submit file
    with open(join(out_dir, "env_vars.txt"), "w") as f:
        f.write("export GITHUB_TAG={}\n".format(github_tag))
        f.write("export GITHUB_TOKEN={}\n".format(github_token))
        f.write("export NUM_JOBS={}\n".format(num_jobs))
        f.write("export WANDB_API_KEY={}\n".format(wandb_api_key))


def get_uuid_from_arg_file(arg_file):
    """ parse the UUID from an args file """
    # todo: eventual goal is to switch all args files to YAML format, and this func would need to be updated
    lines = utils.load_lines(arg_file)
    uuid = None
    for i, line in enumerate(lines):
        if line.startswith("--uuid"):
            uuid = lines[i+1]
            break
    return uuid


def get_log_dir_base_from_arg_file(arg_file):
    """ parse the log_dir_base from an args file """
    # todo: eventual goal is to switch all args files to YAML format, and this func would need to be updated
    lines = utils.load_lines(arg_file)
    log_dir_base = None
    for i, line in enumerate(lines):
        if line.startswith("--log_dir_base"):
            log_dir_base = lines[i+1]
            break
    return log_dir_base


def save_queue_list(out_dir, args_out_dir, queue_list_fn="queue.txt"):
    """ the queue list helps with queuing multiple jobs with specific UUIDs """
    # loop through args files in order, creating an arg_num,UUID list
    args_fns = [join(args_out_dir, x) for x in os.listdir(args_out_dir) if join(args_out_dir, x).endswith(".txt")]
    with open(join(out_dir, queue_list_fn), "w") as f:
        for args_fn in sorted(args_fns, key=lambda fn: int(basename(fn).split(".txt")[0])):
            # write to job num --> uuid map file
            arg_num = int(basename(args_fn).split(".txt")[0])
            my_uuid = get_uuid_from_arg_file(args_fn)
            if my_uuid is None:
                raise ValueError("no UUID detected in argument file {}. currently, all args files must have UUIDs. "
                                 "the reason is because it makes checkpointing and resuming runs a lot easier. "
                                 "also, the submit file and queuing system is currently set up to need UUIDs. "
                                 "if you are generating argument files from a template yaml file, you can specify "
                                 "the UUID 'condor' and this script will take care of generating a new UUID for each "
                                 "generated argument file. in the future, i would like to support not needing a UUID "
                                 "and just generating a random one at runtime. ".format(basename(args_fn)))
            f.write("{},{}\n".format(arg_num, my_uuid))


def create_output_dirs(out_dir, log_dir_base, queue_list_fn="queue.txt"):
    """ creates output directories for each UUID (use queue list to get UUIDs) """
    # create the log_dir_base output directory
    os.makedirs(join(out_dir, log_dir_base))

    # load the queue list (maps job num --> UUID)
    queue_list = utils.load_lines(join(out_dir, queue_list_fn))

    # create output directory for each UUID (use log_dir_name function from train.py)
    for ql in queue_list:
        uuid = ql.split(",")[1]
        os.makedirs(join(out_dir, log_dir_name(log_dir_base, uuid)))


def verify_log_dir_bases(args_out_dir, hardcoded="output/training_logs"):
    """ the log_dir_base is hardcoded to be output/training_logs in the submit file.
        this function verifies that is the case for all jobs
        ideally, we make the submit file more dynamic to support different log_dir_base directories
        however, note that we'd still need all the jobs to have the same log_dir_base
        it just wouldn't need to be output/training_logs """

    args_fns = [join(args_out_dir, x) for x in os.listdir(args_out_dir) if join(args_out_dir, x).endswith(".txt")]
    log_dir_bases = []
    for args_fn in sorted(args_fns, key=lambda fn: int(basename(fn).split(".txt")[0])):
        log_dir_base = get_log_dir_base_from_arg_file(args_fn)
        log_dir_bases.append(log_dir_base)

    # verify that all log_dir_bases are the same
    if not utils.all_equal(log_dir_bases):
        raise ValueError("All argument files must use the same log_dir_base (makes condor runs a lot easier...)")

    # verify that log_dir_bases are equal to the hardcoded output/training_logs
    # todo: support other log_dir_bases besides output/training_logs, requires changing htcondor.sub
    if log_dir_bases[0] is None:
        warnings.warn("Argument files do not specify a log_dir_base, using default from train.py. "
                      "This could cause problems if train.py has a default other than output/training_logs.")
    elif log_dir_bases[0] != hardcoded:
        raise ValueError("Argument files specify log_dir_base={}, but framework currently only supports "
                         "log_dir_base={} for condor runs.".format(log_dir_bases[0], hardcoded))


def parse_kv_pair(pair_str):
    # https://stackoverflow.com/questions/27146262/create-variable-key-value-pairs-with-argparse-python
    if "=" not in pair_str:
        raise ValueError("KEY=VALUE pair does not contain '=' delimiter: {}".format(pair_str))

    pair = pair_str.split("=")
    k = pair[0].strip()
    v = ""
    if len(pair) > 1:
        v = '='.join(pair[1:]).strip()

    return k, v


def parse_kv_pairs(pair_strs):
    d = {}
    for pair_str in pair_strs:
        key, value = parse_kv_pair(pair_str)
        d[key] = value
    return d


def fill_submit_template(env_files_fn, additional_data_files, override_submit_file,
                         save_dir, template_fn="htcondor/templates/htcondor.sub"):

    # load the submit file template as a list of strings (one per line)
    template_lines = utils.load_lines(template_fn)
    # with open(template_fn, "r") as f:
    #     template_str = f.read()

    # first check if we are overriding any submit file lines with override_submit_file
    # if so... implement those first, before filling in env_files and additional_data_files
    # the reason is just in case the user wants to override transfer_input_files
    # they can still have the env_files and additional_data_files filled in by
    # specifying "{transfer_input_files}" in their override value
    if override_submit_file is not None:
        kv_pairs = parse_kv_pairs(override_submit_file)
        for k, v in kv_pairs.items():
            found_existing_line = False
            # check if the template has an existing line for our current override key
            for line_idx in range(len(template_lines)):
                line = template_lines[line_idx]
                # we have an existing line in the template that we need to override
                if not line.startswith("#") and line.split("=")[0].strip() == k:
                    found_existing_line = True
                    template_lines[line_idx] = "{} = {}".format(k, v)
                    break

            # we did not find existing line in the template for our override key
            # add a new line to the beginning of the submit file
            if not found_existing_line:
                template_lines.insert(0, "{} = {}".format(k, v))

    # convert template into a single string for next step where we fill in format args for
    # transferring python environment, additional data files, and any potential future format args
    template_str = "\n".join(template_lines)

    # get list of python environment files that need to be transferred from squid to execute nodes
    # these files could also be transferred from submit node instead of squid if they meet size reqs
    env_files = []
    if env_files_fn is not None:
        env_files = utils.load_lines(env_files_fn)

    # additional data files that should be transferred to execute nodes (from either squid or submit node)
    if additional_data_files is None:
        # if there are no additional data files, make it an empty list
        additional_data_files = []

    # combine all files that need to be added to transfer_input_files
    transfer_input_files = env_files + additional_data_files
    transfer_input_files_str = ", ".join(transfer_input_files)

    # if there is a spot to fill in the transfer_input_files, fill those in
    if "{transfer_input_files}" in template_str:
        template_str = template_str.format(transfer_input_files=transfer_input_files_str)

    with open(join(save_dir, "htcondor.sub"), "w") as f:
        f.write(template_str)

    return template_str


def get_private_tokens(github_token, wandb_api_key):
    """ to help avoid storing GitHub token and wandb api key in files, check env vars for them """

    # if tokens were specified as arguments (not None), those take priority
    # otherwise we check the environment variables
    if github_token is None:
        if "GITHUB_TOKEN" in os.environ:
            github_token = os.environ["GITHUB_TOKEN"]

    if wandb_api_key is None:
        if "WANDB_API_KEY" in os.environ:
            wandb_api_key = os.environ["WANDB_API_KEY"]

    return github_token, wandb_api_key


def prep_run(args):
    # create an output directory for this run
    out_dir = create_run_dir(args.run_name)

    github_token, wandb_api_key = get_private_tokens(args.github_token, args.wandb_api_key)

    # save the arguments for this condor run as run_def.txt in the log directory
    # best practice is to avoid storing GitHub token and wandb api key in a file
    utils.save_args(vars(args), join(out_dir, "run_def.txt"), ignore=["github_token", "wandb_api_key"])

    # generate args files from templates
    args_out_dir = join(out_dir, "args")
    num_jobs = gen_args_from_template_fns(args.args_template_fns, args_out_dir)

    # the GitHub token and WandB API key are stored in env_vars.txt
    # this should probably be changed to avoid storing them in a file
    # some options are to use environment variables or allow user to pass in values on submit node
    save_env_vars(out_dir, args.github_tag, github_token, num_jobs, wandb_api_key)

    # verify log_dir_base (see function for more details)
    hardcoded_log_dir_base = "output/training_logs"
    verify_log_dir_bases(args_out_dir, hardcoded=hardcoded_log_dir_base)

    # save queue list
    save_queue_list(out_dir, args_out_dir)

    # create output directories for each job
    create_output_dirs(out_dir, hardcoded_log_dir_base)

    # fill submit template
    # mainly filling in python environment files and any additional files that need to be transferred to execute node
    fill_submit_template(args.env_files_fn, args.additional_data_files, args.override_submit_file, out_dir)

    # copy over run.sh
    shutil.copy("htcondor/templates/run.sh", join(out_dir, "run.sh"))

    # copy submit scripts to download the code & submit the job
    shutil.copy("htcondor/templates/submit.py", out_dir)
    shutil.copy("htcondor/templates/submit.sh", out_dir)

    # create output directories where jobs will place their outputs
    os.makedirs(join(out_dir, "output/condor_logs"))


def main(args):
    prep_run(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@")

    parser.add_argument("--run_name",
                        help="name for this condor run, used for log directory",
                        type=str,
                        default="unnamed")

    parser.add_argument("--args_template_fns",
                        type=str,
                        help="template files (.yaml) containing arguments to be expanded into multiple args files",
                        nargs="*")

    parser.add_argument("--env_files_fn",
                        type=str,
                        help="path to a text file containing list of Python environment files to be transferred from "
                             "squid to the execute node. could specify those files in additional_data_fns, but"
                             "this allows you to keep track of environment version more easily",
                        default=None)

    parser.add_argument("--additional_data_files",
                        type=str,
                        help="additional data files to transfer to execute node. these will "
                             "get added to transfer_input_files in the HTCondor submit file.",
                        nargs="*")

    # https://stackoverflow.com/questions/27146262/create-variable-key-value-pairs-with-argparse-python
    parser.add_argument("--override_submit_file",
                        metavar="KEY=VALUE",
                        nargs="*",
                        help="set a number of key-value pairs to OVERRIDE the defaults in the template submit file. "
                             "if you are supplying this argument on a command line, do not put spaces before or "
                             "after the = sign, and place double quotes around the value if it has spaces. "
                             "if you are supplying this argument in an args file and calling python condor.py @file, "
                             "then you do not need to place double quotes around the value if it has spaces. "
                             "it is up to your judgement whether to use this argument or create a new template "
                             "submit file. i would use this arg for simple changes like needing more GPUs or "
                             "a different amount of memory. i would create a new template submit file for big "
                             "structural changes like restructuring data flow from submit file to execute nodes.")

    parser.add_argument("--github_tag",
                        type=str,
                        help="GitHub tag specifying which version of code to retrieve for this run",
                        default="master")

    parser.add_argument("--github_token",
                        type=str,
                        help="authorization token for private github repository. if None, script will check "
                             "environment variables for GITHUB_TOKEN and use that if available",
                        default=None)

    parser.add_argument("--wandb_api_key",
                        type=str,
                        help="weights&biases API key. if None, script will check environment variables for"
                             "WANDB_API_KEY and use that if available",
                        default=None)

    main(parser.parse_args())

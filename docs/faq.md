# Frequently Asked Questions

### How is the performance compared to PyEmma?
This depends heavily on the task, but in general it should be comparable or even faster, see [Benchmark](../benchmark/).


### Is there a command line interface?
Yes, indeed. Some useful plots (e.g. CK-test) and tools can be used directly from the command line. Please check out the short tutorial [CLI](../tutorials/cli).


### Is there a shell completion
Using the `bash`, `zsh` or `fish` shell click provides an easy way to
provide shell completion, checkout the
[docs](https://click.palletsprojects.com/en/8.1.x/shell-completion).
In the case of bash you need to add following line to your `~/.bashrc`
```bash
eval "$(_MSMHELPER_COMPLETE=bash_source msmhelper)"
```
In general one can call the module directly by its entry point `$ msmhelper`
or by calling the module `$ python -m msmhelper`. For enabling
the shell completion, the entry point needs to be used.


### Feature X is missing
First, if you are looking for a feature complete package, I would recommand you to take a look at [pyemma](https://github.com/markovmodel/PyEMMA) and [msmbuilder](https://github.com/msmbuilder/msmbuilder). If you believe that a crucial functionality/method is missing, feel free to [open an issue](https://github.com/moldyn/msmhelper/issues) and describe the missing functionality and why it should be added. Alternatively, you can implement it yourself and create a PR to add it to this package, see [contributing guide](../contributing).


### I found a bug. What to do next?
If you find a bug in this package, it is very kind of you to open an issue/bug report. This allows us to identify and fix the problem, thus improving the overall quality of the software for all users. By providing a clear and concise description of the problem, including steps to reproduce it, and relevant information such as device, operating system, and software version, you will help us resolve the problem quickly and effectively. Submitting a [bug report](https://github.com/moldyn/msmhelper/issues) is a valuable contribution to the software and its community, and is greatly appreciated by the development team.

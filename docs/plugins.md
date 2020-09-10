This module is still under development. Code is subject to major changes.*

# VFRAME Plugins

Add plugins to extend VFRAME. They're ideal for focused research projects and to help keep projects organized when lots of customized scripts are needed. Write as many CLI scripts as you want and add them to a custom plugin.

There are 2 types of plugins: standalone/utility and pipe. Pipe refers to the image processing pipe that you can use to use multiple commands together. Standalone scripts are self-contained.

## Add a VFRAME Plugin

This section is under development.

## Create a New Plugin

The follow example will create a new plugin called "vframe_new_plugin" with 2 command scripts.

First, clone the VFRAME plugin template into the plugins directory:

```
cd vframe/vframe_cli/plugins/
git clone https://github.com/vframeio/vframe_template_plugin
```

MOcreate a new directory inside the `vframe_cli/plugins` directory. For example, to create a plugin for your custom visualization scripts called "vframe_new_plugin":

```
# Tree structure of "vframe_new_plugin"
vframe_new_plugin/
    - docs/
    - README.md
    - LICENSE.md
    - new_plugin 		<-- source code
    	- commands/ 	<-- .py commands
    	- settings/		<--- settings
    	- utils/		<-- utility scripts
```

Next, add a few scripts:
```
commands/my_script.py
commands/my_other_script.py
```

Then, activate your new new plugin by adding it to the plugins configuration YAML file:

```
- name: new_plugin
  plugins:
    - commands: plugins/vframe_new_plugin/new_plugin/commands/

```

You should be able to run your script from the main CLI with a command similar to:

```
./cli.py new_plugin my_script
```

The directory structure of `vframe_new_plugin/new_plugin/` is, for now, verbose but it helps avoid namespace conflicts. It is automatically added to the the Python paths when running the base CLI script. Access your plugin scripts as `from my_plugin.utils import my_utils` and reference other vframe plugins or core imports by using `from vframe.utils import file_utils`. Plugins can reference other plugins.


## Multiple Deploy Keys on Github.com

An imperfect but usable way to access multiple deploy keys on github:

```
# generate a new key
# and save it to ~/.ssh/id_rsa_vframe_custom_plugin
# then add the id_rsa_vframe_custom_plugin.pub key to github deploy keys
ssh-keygen -t rsa

# use modified github ssh command to reference this key
GIT_SSH_COMMAND='ssh -i /home/ubuntu/.ssh/id_rsa_vframe_custom_plugin-o IdentitiesOnly=yes' git clone git@github.com:youruser/vframe_custom_plugin

# pull
GIT_SSH_COMMAND='ssh -i /home/ubuntu/.ssh/id_rsa_vframe_custom_plugin-o IdentitiesOnly=yes' git pull

 ```
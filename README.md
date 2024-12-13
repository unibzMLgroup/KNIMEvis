# knimeVizLab

![alt text](LogoProject.png)

## Description
This repository serves as the official [UNIBZ](www.unibz.it) project for creating custom KNIME nodes dedicated to image processing using Python. The goal is to provide seamless integration between KNIME's workflow analytics platform and Python's advanced capabilities for image analysis and processing.

With this project, users can:

- Develop custom KNIME nodes tailored to specific image processing requirements.
- Leverage Python libraries and tools for advanced image manipulation and analysis.
- Simplify workflows by combining KNIME's no-code/low-code environment with Python's scripting flexibility.


## Installation

To start developing KNIME nodes with Python, follow these steps:
1. **Clone this repository**:
    ```bash
    git clone ...
    ```

2. **Install the KNIME Analytics Platform**
    Install [KNIME Analytics Platform](https://docs.knime.com/2024-12/analytics_platform_installation_guide/index.html#_installing_knime_analytics_platform) version 4.6.0 or higher.
    Make sure to also install the KNIME Python Extension Development (Labs). In case of problem follow the offical guide line on how to develope pure [python nod in KNIME](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#extension-bundling)

3. **Set up a Conda/Python environment**
    Create a conda/Python environment containing the [knime-python-base metapackage](https://anaconda.org/knime/knime-python-base), together with the node development API [knime-extension](https://anaconda.org/knime/knime-extension) for the KNIME Analytics Platform you are using

    ```bash
        cd KnimeVisLab
        conda env create -f env.yml
    ```

4. **Modify the config.yml file**
    Update the following fields in the `config.yml` file:
        - Set the absolute path to point to the src directory of this repository.
        - Set the path to the Conda environment you just created.

5. **Update the knime.ini file**
    Locate the `knime.ini` file in the KNIME installation directory and sdd the following line to the end of the file:

    ```-Dknime.python.extension.config=path/to/config/config.yml```

[comment]: # TODO remove from the official repo
## [Bundling a Python extension](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#extension-bundling)

Once you have finished implementing your Python extension, you can bundle it, together with the appropriate conda environment, into a local update site. This allows other users to install your extension in the KNIME Analytics Platform.

Follow the steps of extension setup. Once you have prepared the YAML configuration file for the environment used by your extension, and have set up the knime.yml file, you can proceed to generating the local update site.

We provide a special conda package, knime-extension-bundling, which contains the necessary tools to automatically build your extension. Run the following commands in your terminal (Linux/macOS) or Anaconda Prompt (Windows). They will setup a conda environment, which gives the tools to bundle extensions. Then the extension will be bundled.
By default, the conda environment will bundle the extension for the latest KNIME Analytics Platform version. If you want to bundle the extension for a specific KNIME version, you have to install the corresponding conda package. You can specify the version when you create the environment , e.g. knime-extension-bundling=5.4. When building an older version, the environment YAML files must contain the corresponding versions of the knime-python-base and knime-extension packages, e.g.- knime-python-base=5.3 when bundling for version 5.3.

1. **Create a fresh environment** prepopulated with the knime-extension-bundling package:

    ```
    conda create -n knime-ext-bundling -c knime -c conda-forge knime-extension-bundling=5.4
    ```

2.   **Activate the environment**

    ```
    conda activate knime-ext-bundling
    ```

3. **Run bulding script**    

    With the environment activated, run the following command to bundle your Python extension:

    _macOS/Linux_:
    
    ```
    build_python_extension.py <path/to/directoryof/myextension/> <path/to/directoryof/output>
    ```
    _Windows_:
    ```
    build_python_extension.bat <path/to/directoryof/myextension/> <path/to/directoryof/output>
    ```
    where <path/to/directoryof/myextension/> is the path to the directory containing the knime.yml file, and <path/to/directoryof/output> is the path to the directory where the bundled extension repository will be stored.

    Further instructions are given by build_python_extension.py --help (macOS, Linux) or build_python_extension.bat --help (Windows) and will be outlined upon execution of the script.
    
The bundling process can take several minutes to complete. 

## How to use the builded KNIME ext


1. Add the generated repository folder to KNIME AP as a Software Site in File → Preferences → Install/Update → Available Software Sites

2. Install it via File → Install KNIME Extensions

## Usage
Once you have completed the installation, you can begin developing and using custom KNIME nodes for image processing. To integrate your Python-based node into KNIME, follow these steps:

1. Create your custom node in Python.
2. Integrate the node into your KNIME workflow by using the custom node created in the previous step.
3. Execute the workflow, and the image processing will be handled by Python scripts through KNIME's interface.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
We welcome contributions to this project! To contribute:

- Fork this repository.
- Create a new branch.
- Make your changes and commit them.
- Submit a pull request.

Please make sure your code follows the existing style and includes appropriate tests.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

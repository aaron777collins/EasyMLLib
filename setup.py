from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = 'EasyMLLIB'
LONG_DESCRIPTION = 'EasyMLLIB - A library to make machine learning easier'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="EasyMLLIB",
        version=VERSION,
        author="Aaron Collins",
        author_email="aaron777collins@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['certifi','charset-normalizer','click','colorama','cycler','et-xmlfile','Flask','fonttools','idna','itsdangerous','Jinja2','joblib','kiwisolver','MarkupSafe','matplotlib','numpy','openpyxl','packaging','pandas','Pillow','pyparsing','python-dateutil','pytz','requests','scikit-learn','scikit-plot','scipy','six','threadpoolctl','urllib3','Werkzeug'], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'EasyMLLIB', 'Machine Learning'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
        ]
)

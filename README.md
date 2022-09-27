# PyPose tutorials

# 1. Contributing to Documentation

## 1.1 Build docs locally

1. Sphinx docs come with a makefile build system. To preview, build PyPose locally and

```bash
pip install -r requirements.txt
make html
```

2. Then open the generated HTML page: `_build/html/index.html`.

3. To clean and rebuild the doc:
```
make clean
```


## 1.2 Writing documentation

We use sphinx-gallery's [notebook styled examples](https://sphinx-gallery.github.io/stable/tutorials/index.html) to create the tutorials. Syntax is very simple. In essence, you write a slightly well formatted python file and it shows up as documentation page.

Here's how to create a new tutorial or recipe:
1. Create a notebook styled python file. If you want it executed while inserted into documentation, save the file with suffix `tutorial` so that file name is `your_tutorial.py`.
2. Put it in one of the beginner_source, intermediate_source, advanced_source based on the level. If it is a recipe, add to recipes_source.
3. For Tutorials (except if it is a prototype feature), include it in the TOC tree at index.rst
4. Create a pull request.

In case you prefer to write your tutorial in jupyter, you can use [this script](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe) to convert the notebook to python file. After conversion and addition to the project, please make sure the sections headings etc are in logical order.


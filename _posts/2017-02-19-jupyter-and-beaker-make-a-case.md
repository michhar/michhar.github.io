---
layout: post
title: The Notebook Superhero -- Is It Always a Contest?
comments: true
description: A discussion around Beaker and Jupyter notebooks
image:  /img/jupyter_beaker/header_beaker_n_jupyter.jpg
tags:
    - r
    - python
    - notebooks
    - jupyter
---

[**tl;dr**](https://en.wikipedia.org/wiki/TL;DR):  Once you learn how to use one kind of notebook system, the knowledge will transfer easily to another.  Here, we're discussing two, Jupyter and Beaker.  Because Jupyter is much more mature of a project, it'd probably be the best place to start.  But for those with that extreme sense of adventure and/or use Python 2 a lot with other languages, give the Beaker notebooks a chance.  Loads of potential for data scientists there.

![is there a wall between jupyter and beaker]({{ site.baseurl }}/resources/images/jupyter_beaker/header_beaker_n_jupyter.jpg)

### What's a notebook?

Have you ever taught programming and wished to have the class notes, sample code and exercises with instructions all in one place?  Have you ever heavily commented your code and wished it was more readable?  Have you used R Markdown and wished to run individual code chunks with only a button or keyboard shortcut?  Have you ever wished to use multiple programming languages in the same place, same document?

So, the story begins with my wish for a better way.  When I discovered notebooks, at first, I felt strange programming in a browser until I discovered I could annotate the code with pleasant, easy-to-read text and for some reason that opened up a whole world.  I began documenting my research work more, creating clear and rich teaching aids, and enhancing my work to share with others in a reproducible way or at least with clear instructions and notes in nice looking text rather than sometimes hard-to-read comments within the code (which I still do of course).  It was the annotations that made it worth my time to learn.

There are several notebook systems out there and they all seem to behave, at their core, the same way in that I can run interactive code cells and document my work in pleasant-to-read formats.  They do vary in their use cases, such as RStudio's notebook being more geared towards the R programmer (although it has extension packages for other languages now) or the Beaker notebooks for combining multiple languages into a workflow.  A sample of notebook "providers" is as follows.

* [Beaker](http://beakernotebook.com/)
* [Zeppelin](https://zeppelin.apache.org/)
* [Spark](http://spark-notebook.io/)
* [Jupyter](http://jupyter.org/)
* [Jupyterlab](https://github.com/jupyterlab/jupyterlab) (preview)
* [RStudio](http://rmarkdown.rstudio.com/r_notebooks.html)
* and more


### Introducing our players

**Beaker**, a polyglot notebook system, is based on IPython (amongst other things) and **Jupyter**, supporting over 40 programming languages, is based on IPython (amongst other things).  They both allow multiple languages from within the same notebook and both run on top of Python.  I found I was able to install either one without the command line so they seemed pretty easy to get going on (Jupyter did require one command in the terminal to start which was a simple task).

They are both open source projects and being built on IPython have similar notebook interfaces so it'd be easy to switch over from one to another once you get the hang of notebooks.

Differences and distinguishing factors discussed below.

### Jupyter:  customizable and sometimes magic 

![jupyter with a grade3 theme]({{ site.baseurl }}/resources/images/jupyter_beaker/jupyter_grade3theme.png)
<p align="right"><b>A Python 3 flavored Jupyter notebook with a "grade3" theme (theme from Kyle Dunovan's jupyter-themes repo)</b></p>

The Jupyter project is much more mature than the Beaker project and thus has a reliable and almost totally bug-free experience (nothing is completely bug-free).  It's pretty much the classic notebook system, but gives the us the ability to use it for reproducible research, publish papers, do presentations with live code, create blogs (not this one, although it's in markdown at least), and the list goes on.  It's a mature project with many add-ons and features available.

Returning to basecamp, Jupyter notebook setups can be simple and basic, one language supported and the basic theme, or much more complex, supporting several languages chosen from a drop-down menu and having extensions to check spelling and perhaps a custom theme to pretty it up.  Out of the box, they simply work with the default Python.  It's a very transparent system.  What you add on is done by you, but you must take care of what that add-on requires.  Jupyter notebooks are meant to be simple, useful and clean (I've seen and made many of messes so I aim for this).

An Anaconda install gives us Jupyter notebooks automatically.  Whichever is the default Python, becomes the default Python version for the notebook (basically whatever is first in our PATH if we have 2 and 3).  We could also install with `pip`, Python's package manager.

Jupyter, when using the python kernel, can incorporate "magics" or other languages within the same notebook (and sometimes passing variables back and forth like with R kernel and rpy2 or javascript).  Some cell magics are listed here (the "%%" is literally the syntax we use in a notebook cell to designate):  

* %%fortran
* %%cython
* %%javascript
* %%html
* %%bash
* %%latex
* %%perl
* %%python2
* %%python3
* %%ruby
* %%R
* others (incl. incubator projects like [sparkmagic](https://github.com/jupyter-incubator/sparkmagic), which created magics within it, in the context of working with spark clusters)


These languages, of course, must be on the system hosting the notebook.  In addition, the Jupyter project reports over 40 languages supported, but this does not mean they all have magics and can be run from an IPython notebook (IPython, here, referring to the Python kernel, but it can also refer to a previous Python notebook project).  Also, custom kernels for languages not supported can be made according to the Jupyter docs.

One customization I really love is `nbextensions` (more [here](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)) which adds functionality to a Jupyter notebooks such as a table of contents, section numbering, highlighting, and spellcheck to name a few.  Personally, I found the TOC and spellcheck very, very useful as I get lost easily and spell quite horribly.

Another customization is around adding a theme, but more on that below.

A really nifty meta-feature is that GitHub renders static versions of IPython/Jupyter notebooks (.ipynb files which are just JSON) which makes viewing your work and the work of others very easy from GitHub.

You can find the Jupyter project on GitHub at [https://github.com/jupyter/notebook](https://github.com/jupyter/notebook).

### Beaker:  a true polyglot

![beaker screenshot]({{ site.baseurl }}/resources/images/jupyter_beaker/beaker_d3.png)

<p align="right"><b>Python 2 and JavaScript with D3 sharing variables (entire code sample is in the D3 Demo notebook that comes with Beaker)</b></p>

A Beaker notebook is different from a Jupyter notebook in that it can easily pass data from one cell to the next even if the code in each cell is in a different programming language.  This is the **big** selling point of Beaker notebooks.  Literally, we can share one variable from Python to Javascript, for example, by just prefacing it with `beaker.`.  Woah.  This opens up a realm of possibilities.

Beaker notebooks give us more default functionality and ease-of-use than a Jupyter notebook at the expense of being less transparent.  If all you need is Python 2, they are super easy and very user-friendly.  Also, Beaker starts up with tons of sample code, called Demos, at your fingertips for most if not all of the supported languages.

Beaker, so far, out-of-the-box, supports* 17 languages:

* Clojure
* C++
* Python2 (called IPython by Beaker)
* Python3
* R
* SQL
* JavaScript
* Scala
* Node
* Torch
* Julia
* Java
* Kdb
* Groovy
* HTML
* Ruby
* TeX

\*_You still need to have the backend interpreter or compiler (just like in Jupyter) and certain plugins in most cases to connect it up to Beaker._

On the origins of Beaker, in their own words... 

> Beaker is built on many fantastic open source projects including Angular, Bootstrap, CometD, Gradle, Guice, IPython, Jackson, Jetty, Nginx, Rserve, and others for which we are very grateful.

Beaker, too, is open source as a "base" or something to run locally, host oneself, or use with Docker.  You can check it out on their GiHhub repo at:  [https://github.com/twosigma/beaker-notebook](https://github.com/twosigma/beaker-notebook).

Beaker has better notebooks management features (such as listing your open notebooks with time stamps).  The UI looks a bit nicer as well.  

> Aside:  Those who like to see their files listed, however, should try Jupyterlab which feels more like RStudio than a notebook system.  It has nice management features as well, but more around transparency into the file system and has the ability to open different file formats plus a classic interpreter.  It's out of the scope of this post for sure.

And some may not see this tiny little note in a screenshot of a guide for what you can put in the text parts right on their GitHub readme, but they totally mention Donald Knuth, one of my favorite people ever.  +1.

### Installing it

Neither Beaker, nor Jupyter, require the command line for installation.  An install of Anaconda for Python includes Jupyter notebooks.  To run it however, one will need to type `jupyter notebook` from the command line, but that's really it (you can also install Jupyter from the command line with `pip`). 

The tricky part sometimes for Jupyter is getting other kernels (support for other languages) installed.  But my other _de facto_ language is R and I simply used the conda compatible [R-Essentials](https://anaconda.org/r/r-essentials), which gives me the R kernel option (Yay!!) (and by far the easiest way to get the R kernel working that I've found - see [this](https://www.continuum.io/blog/developer/jupyter-and-conda-r) blog for more on R-Essentials).  I gave up getting R to work in Beaker after toying around for an hour or so (granted that wasn't a long time and smarter folks could probably get it working) running up against an undocumented R package called Rserve, a dependency for R in Beaker.

It appears Beaker by default expects, as it [says here](https://github.com/twosigma/beaker-notebook/wiki/Python-Mismatch-Errors), a Python 2 conda install (which is weird I thought due to Python 2 becoming a legacy language soon).  So, when I tried it with my Python-3-only conda install, I had bad luck running an IPython cell, although Python 3 cells worked.  I did solve the IPython cell issue according to some pretty easy-to-follow advice on their wiki about specifying the IPython path and adding a Python path pointing to a Python 2 install (in a config file called `beaker.pref.json`).  Beaker's wiki is, in general, very helpful I've found thus far.


### Themes:  design can win people over

![]({{ site.baseurl }}/resources/images/jupyter_beaker/jupyter_basic_js.png)
<p align="right"><b>A Jupyter notebook with the `mm_dark_theme` (theme by Milos Miljkovic and found on GitHub)</b></p>

Out of the box, we get a couple of themes for our notebooks with Beaker,  Default and Ambiance.  With Jupyter, we can add any theme we'd like or like to create in a few ways, but my favorite and simplest is just adding a `custom.js` and `custom.css`  to a folder called `~/.jupyter/custom/` (create one if it's not there).  The `~` refers to the user's base directory, often used in Unix systems. 

For the Jupyter notebook above, I used a dark theme which is found [here](https://github.com/miishke/mm_dark_theme), but you can find them all over place on GitHub.  The installs will vary so I stick with the `custom.js` from the `mm_dark_theme` project/repository and just switch out the css files (probably not the best practice mind you, but I always give credit to the creators of the css, or stylesheets).

Yes, the Jupyter method to include a theme is one or two steps more complicated, but it's truly custom and themes can be important for various reasons, like emphasizing the graphs and text to non-technical folk in a more pleasant background color than white and nicer fonts.  I do have to admit I was pleasantly surprised by having a theme choice in Beaker and then how easy it was to switch them.

### Competition or no?

So, is it a competition?  I'd say not so much given we don't really have an apples to apples situation.  While Beaker may be trying to fill the gaps or make its niche, I'd say it's in fact creating a whole new experience for data scientists which could be extraordinary if they'd only make some small adjustments (such as make the deployment Python version agnostic).  I hope these two projects, and the others, continue to complement each other and grow even better.

### Questions I have, but didn't answer here

* Do I really need to pass variables from one language to another or are magics in Jupyter sufficient for incorporating the same languages into one notebook?
* Why would I choose Jupyterlab over Jupyter notebooks?  Is it a step forward or simply a divergence?
* RStudio does similar things these days and with XRPython we can embed python in a notebook-eqsue environment — and might I add tidyverse.  Why not use that?
* Which one is the best for Spark?

Thanks for reading.

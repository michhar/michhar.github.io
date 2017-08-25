---
color:  lime
width:   6
height:  3
layout: post
title: Javascript and Python Meet through Magic and IPython
comments: true
description: A short, simple post on passing variables between these two languages in Jupyter notebooks
cover:  /img/js_python_friends/header_pic.jpg
tags: [python, javascript, jupyter]
---

### A little exchange

_Python_:  "I want your silly game to have a starting value of 100 for life points for all players."

_Javascript_:  "I read you loud and clear.  Let's take that starting life and play this silly game.  Alfred and Wallace are on the same side battling orcs.  Alfred decides to give life to Wallace because Wallace appears to be about to bravely charge onto the battle field and may need it.  Done.  That's all the game can do right now because I just started making it."

_Python_:  "Let's see...oh wow.  Your game code actually worked.  Alfred now has 99 life points and Wallace has 101.  Good job us."

I went on a hunt to find the cleanest and most succinct way to pass a variable from Python to Javascript and then back to Python in a Jupyter notebook.  I wanted a proof of principle upon which I could later base my D3 graphics using Python data as `pandas` dataframes.

So, this silly example can be translated into code as follows.

First, things are actually easy for us because we are in a Jupyter notebook living in a browser, utilizing all sorts of widgets and HTML elements already.  We can tap into that (and actually the DOM) to get some of the functionality we require.

So, in Python we can use the IPython.display module with the HTML function allowing us to embed an element for later use.

```python
# Python
from IPython.display import HTML

pystartlife = str(100)
HTML("<div id='textid'>" + pystartlife + "</div>")
```

We've created a Python variable, `pystartlife`, and embedded it as a `div` element, literally just using raw HTML.  Now we use the Javascript magics (`%%javascript`) to create a Javascript coding environment for the next cell.  (BTW there are magics for many more languages - very cool.)  

In the Javascript, now, we grab the `div` element with the Python variable from the document or webpage and play our game.  We also write the game.  (This game is based on a [Node.js tutorial](https://www.youtube.com/playlist?list=PL6gx4Cwl9DGBMdkKFn3HasZnnAqVjzHn_) by thenewboston on YouTube - thank you Bucky!).

```javascript
%%javascript

// Get the python variable from the DOM
var startlife = document.getElementById('textid').innerHTML;

// Define a User class with a method
function User() {
    this.name = '';
    this.life = Number(startlife);
    this.giveLife = function giveLife(targetPlayer) {
        targetPlayer.life += 1;
        this.life -= 1;
    }
}

// Use class
var Alfred = new User();
var Wallace = new User();

// Names were blank so give them name values
Alfred.name = 'Alfred';
Wallace.name = 'Wallace';

// Let's play a game!

// Let Alfred give life to Wallace
Alfred.giveLife(Wallace);

// Save these variables back to python variables to work with later
IPython.notebook.kernel.execute('Alfred_life="' + Alfred.life + '";');
IPython.notebook.kernel.execute('Wallace_life="' + Wallace.life + '";');
```



We grab the Python variable now embedded with `document.getElementById('idname').innerHTML`, a DOM function and attribute which grabs the `div` by its _id_ (not name) and takes what is in the text part of the `div`.

Then we define a class in Javascript that contains an altruistic method for giving life.  We create some users and give them names.  We then "play" the game and allow one to give life to another with the method in the User class - how generous of Alfred!

Lastly, the lovely magic part, we use this sneaky Javascript function from the IPython class that executes Python statements.  We could execute any Python statement really in this way, e.g., `IPython.notebook.kernel.execute('print("Hello World!")');`  We include our life values (Javascript variables) into this executable string and, well, execute it.

And to see if the magic worked, we run a simple couple of print statements back in Python:

```python
# Python
print(Alfred_life)
print(Wallace_life)
```

With low and behold and output of:  99 and 101.  Way to pay it forward, Alfred!


You can find this code and more in [this](https://github.com/michhar/python-jupyter-notebooks/blob/master/general/Jupyter_and_JavaScript.ipynb) Jupyter notebook.



---
layout: post
title:  Overlaying a Website ontop of a GitHub Repository
comments: true
description:  A quick discovery post on using static site generators to overlay a website on top of a GitHub repo for displaying docs, portfolios/products, and blogs.
tags: [web-dev]
highlight: true
cover: /img/sitegenpost/octocat_web_mharris.png
---

**tl:dr**:  A quick discovery post on using static site generators to overlay a website on top of a GitHub repo for displaying docs, portfolios/products, and blogs.

I recently discovered I can have a static website for each of my GitHub repositories, which seems like overkill for 35+ repos.  That being said, it certainly would be nice to attractively present some of my content I've painstakingly written in Markdown files to showcase my work.

Using static site generators, like Jekyll or MkDocs (and others — good article [here](https://www.fullstackpython.com/static-site-generator.html)) combined with a templating engine like Jinga, opens up a lovely world of having human-friendly interfaces on top of my repos that I want to share.  The complexity of the sites can get quite intricate — to my hearts content.

![My personal webjeda cards site](/img/sitegenpost/my_webjeda_main.png)
<p align="right"><b>Usage of the Jekyll Webjeda theme for a course listing site</b></p>

Markup languages like Markdown are pretty easy to write in and create text files with headings, lists, tables and such (GitHub made a nice guide [here](https://guides.github.com/features/mastering-markdown/)).  However, if you like reStructuredText or HTML better there are generators out there for you.

There's a great site to shop for a Jekyll theme (where I began this journey), [here](http://jekyllthemes.org/).  They are mostly slanted towards blog writers as that was the reason for the genesis of Jekyll by the founder of GitHub (thanks Tom Preston-Werner!  Find out more about Jekyll on [this](https://www.smashingmagazine.com/2015/11/static-website-generators-jekyll-middleman-roots-hugo-review/#jekyll) blog).

There is of course the use of GitHub Pages to simply render the repo's README markdown file by clicking on **Settings** and scrolling down to **GitHub Pages**, then selecting a theme directly at that point.  

Basically, I'm just scratching the surface here on another way.  Also, the generators I chose all look good on hand-held devices - an important aspect to consider.

These are some scenarios for the site generators I'm introducing based on research and actual work I've done:

* **Chalk**
  * Blogs
  * Things listed by dates

* **Webjeda Cards**
  * Blogs
  * Portfolios
  * Modular presentations e.g. product cards

* **MkDocs**
  * Documentation
  * Guides or tutorials

With the following information, hopefully you can begin to successfully build sites based on these generators.  I hope to fork them at some point for contributing back and encourage you to do so if you figure out something useful.

Clarification on project setups:  some themes or static site generators that use repositories for content, have a branch for building the site (source) and a branch (usually called `gh-pages`) for the sites deployment files (MkDocs sites do this).  Some themes just work under `master` or have a mirrored dev branch.

### Chalk

![Chalk screenshot](/img/sitegenpost/chalk_main.png)
<p align="right"><b>Chalk demo site</b></p>

_Chalk is a high quality, completely customizable, performant and 100% free blog template for Jekyll._

Creator's Profile:  [https://github.com/nielsenramon](https://github.com/nielsenramon)

Chalk is my favorite Jekyll theme for blogging.  It's simple and clean in its look.  However, Chalk doesn't support the standard way of working with Jekyll on GitHub pages due to custom plugins.  There's a little bit more complexity around building the site with these plugins, but all of the scripts are provided so it's actually quite easy in the long run to build and deploy to GitHub Pages or another hosting service.

Check out these sites:

[Demo site](http://chalk.nielsenramon.com/)

[Chalk GitHub repo](https://github.com/nielsenramon/chalk)

Recommended tweaks: 
 
*  Use the dark code highlighting theme inside of the light Chalk theme for a Sublime Editor-like effect.  Modify an import at the bottom of the `/_assets/stylesheets/light.scss`:

    ```
    @import "modules/highlights-dark";
    ```
    
*  Add an icon to the post listing page (main page) by editing the `index.html` with a `div` tag:

    ```html
    <div class="article-list-tags">
      <img src="{{ post.logo }}" alt="post logo" 
          href="{{ post.url | relative_url }}" 
          width="30%" align="center"></img>
    </div>
    ```
    And adding the `post.logo` to the YAML heading on the post as in:
   
   ```text
    ---
	...
	comments: true
	description: A short chatterbot dev story
	logo:  "../resources/images/ocrbot_local.png"
	tags:
	...
	---
   ```

![Chalk screenshot my site](/img/sitegenpost/my_chalk_main.png)
<p align="right"><b>A tweaked Chalk-based site with icons on main page</b></p>

### Webjeda Cards

![Webjeda screenshot](/img/sitegenpost/webjeda_main.png)
<p align="right"><b>Webjeda demo site</b></p>

_Webjeda Cards is a Bootstrap based jekyll theme for portfolio, photography or any kind of blog._

Creator's Profile:  [https://github.com/sharu725](https://github.com/sharu725)

This Jekyll theme is fantastic for more modular postings (like course or product listings).  It could also be a great place to show off your work like photos you've taken with a nice write-up.  Perhaps, this could be the main page of your site and you could link to all of the repos you'd like to share from this card layout design.

Setup note:  you can of course simply use the master branch here for building _and_ deploying, but it's always nice to have a separate branch we often see called `gh-pages` for the sites actual deployed content.

Check out these sites:

[Demo site](http://webjeda.com/cards/)

[Webjeda Cards GitHub repo](https://github.com/sharu725/cards)

Recommended tweaks:

* In `index.html` added a variable to automatically pull in the "Read" button name (so, instead of just "Read" it's a custom label on the button).  This is done with the `post.label`:
	
	```html
	 <div class="panel-body">
	 <h3 class="panel-title pull-left">{{ post.title }}
	    </h3><span class="post-meta pull-right"><small></small></span>
	 <a href="{{ post.url | prepend: site.baseurl }}" 
	    class="btn btn-primary btn-sm pull-right mt10">
	    Go to {{post.label}}</a>
	 </div>
	```
	
* Then, in the post you'll have a YAML header with a `label` variable like:
	
	```text
	---
	layout: post
	title:  Microsoft R Server and SQL Server R Services Labs
	categories: mrs
	img: hikeclouds.jpg
	label: Labs
	resource:
	---
	```

### MkDocs

![MkDocs screenshot](/img/sitegenpost/my_mkdocs_main.png)
<p align="right"><b>Docs for one of my projects using the readthedocs theme</b></p>

_MkDocs is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation._

Creators' profile:  [https://github.com/mkdocs](https://github.com/mkdocs)

This is a great alternative to a Sphinx build for a readthedocs style documentation page.  It's very easy to setup and use.  There are other builtin themes as well.  It's fairly pre-baked, but very good for what it's good for.

[Documentation for MkDocs](http://www.mkdocs.org/#mkdocs)
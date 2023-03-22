# DarkWeb Graph Analysis 🌐

<p align="center"> 
  <img src="https://user-images.githubusercontent.com/91635053/226994382-dc066ac4-dcb8-4a14-b032-593e204d79b8.png" alt="Network Image" height="300px" width="300px">
</p>


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents">Table of Contents :book: </h2>

  <ul>
    <li><a href="#overview">Overview</a>:eyes:</li>
    <li><a href="#project-folders-description">Project Folders Description</a>:floppy_disk:</li>
    <li><a href="#graph-analysis">Graph Analysis</a>:globe_with_meridians:</li>
    <li><a href="#references">References</a>:book:</li>
  </ul>

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png"/>
</div>
<!-- OVERVIEW -->
<h2 id="overview">Overview :eyes: </h2>

<p align="justify"> 
  Graph theory has long been a favored tool for analyzing social relationships as well as quantifying
  engineering properties such as searchability. For both reasons, there have been numerous graph-theoretic
  analyses of the World Wide Web (www) from the seminal to the modern. These analysis have been repeated for the notorious “darkweb”. The darkweb is sometimes loosely    defined as “anything seedy on the Internet”, but we define the darkweb strictly, as simply all domains underneath the “.onion” psuedo-top-level-domain, i.e., we define the darkweb to be synonymous with the onionweb.
</p>
The graph represents the DarkWeb network: <br>
<ul>
<li>A node represents a domain. </li>
<li>An edge (u to v) means there exists a page within domain u linking to a page within domain v.</li>
<li>The weight of the edge from u to v is the number of pages on domain u linking to pages on domain v.</li>
</ul>

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png"/>
</div>

<!-- PROJECT FILES DESCRIPTION -->
<h2 id="project-folders-description">Project Folders Description :floppy_disk: </h2>

<ul>
  <li><b>Dataset</b>: in this folder there are some useful csv files:
  <ul>
  <li>Darkweb.csv: this is the darkweb graph, which is composed by 4 column (Source, Target, Type, Weight)
  <li>Darkweb_scc.csv: the same as before with some added info like Id and Label (exported from Gephi)
  </ul>
  </li>
  <li><b>Gephi</b>: in this folder there is Gephi project that you can import in the homonym software
  <li><b>SourceCodePython</b>: in this folder there is Python source code that is necessary to run most project analysis
  <li><b>SourceCodeMatlab</b>: in this folder there is Matlab source code that is necessary to run community analysis, especially all what concerns persistance probabilities
  <li><b>Results</b>: in this folder you can find all results carried out by the whole analysis process. Results from Gephi, Python and Matlab
  <li><b>Presentation</b>: in this folder you can find the pptx file related to the project presentation
  <li><b>Paper</b>: in this folder you can find the original paper related to this project in which my analysis is based on
</ul>

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png"/>
</div>

<!-- Analysis -->
<h2 id="graph-analysis">Graph Analysis  :small_orange_diamond: </h2>
Some initial stats:
<br>
<div align="center">

| | |
|:-------------------------:|:-------------------------:|
|<img src="https://user-images.githubusercontent.com/91635053/227004581-964b0927-2d24-4e10-8177-5a903ddbc93a.png" alt="Network Image" height="282px" width="380"> |  <img src="https://user-images.githubusercontent.com/91635053/227004630-21b7807e-d9c8-42e4-b753-46a4595b6a04.png" alt="Network Image" height="282px" width="380"> |

</div>

Obviously, the DarkWeb is a very danger zone: so in this project what I would to point out is understanting what is the potential way to destroy it.
We can analyze what is the network response regarding failures (random attacks) and targeted attacks: the metrics taken into account are the size of the largest connected component and the efficiency value.

<div align="center">


| | |
|:-------------------------:|:-------------------------:|
|<img src="https://user-images.githubusercontent.com/91635053/227018847-52592976-d9d1-489a-bd43-2bece0e41cbf.png" alt="Network Image" height="282px" width="360"> |  <img src="https://user-images.githubusercontent.com/91635053/227018984-435aadaf-85b8-416c-9546-10d2e490e87d.png" alt="Network Image" height="282px" width="360"> |

</div>

If you want to deep into the whole analysis you are free to explore the entire repo. Have fun! :fireworks:

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png"/>
</div>

<!-- Analysis -->
<h2 id="references">References :small_orange_diamond: </h2>

[1] Dataset DarkWeb (<a href="https://icon.colorado.edu/#!/networks">Dataset DarkWeb</a>)
<br>
[2] Graph Theoretic Properties of the Darkweb, V. Griffith, Y. Xu, C. Ratti, 2017. (<a href="https://arxiv.org/pdf/1704.07525.pdf">arxiv.org</a>)
<br>
[3] Finding and testing network communities by lumped Markov chains, C. Piccardi, 2011. (<a href="https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0027028">journals.plos.org</a>)
<br>
[4] Profiling core-periphery network structure by random walkers, F. Della Rossa, F. Dercole, C. Piccardi, 2013. (<a href="https://www.nature.com/articles/srep01467">nature.com</a>)
<br>

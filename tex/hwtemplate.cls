\ProvidesClass{hwtemplate}[2017/07/10 ver.1.0 template class]

% Font size and paper type
\LoadClass[a4paper, 10pt]{article}     
\usepackage{geometry}       
\geometry{a4paper}
\geometry{margin=1in} 
\usepackage{paralist}
  \let\itemize\compactitem
  \let\enditemize\endcompactitem
  \let\enumerate\compactenum
  \let\endenumerate\endcompactenum
  \let\description\compactdesc
  \let\enddescription\endcompactdesc
  \pltopsep=\medskipamount
  \plitemsep=1pt
  \plparsep=1pt
\usepackage[english]{babel}
\usepackage[utf8]{inputenc}


%-------------------------------------------------------------------------------
% Math, table and graphics
%-------------------------------------------------------------------------------

\usepackage{bbm, bm, amsmath, amssymb, amsthm, mathrsfs}
\usepackage{booktabs, tikz, float}

% Required for boldface (\bf and \bfseries) tabular columns
\usepackage{array} 

% multirow tables
\usepackage{multirow}
\usepackage{makecell}
\usepackage{caption} 
\captionsetup[table]{skip=2pt}

% Required for tabular columns alignment
\usepackage{tabularx} 

% Required for ifthenelse statements
\usepackage{ifthen}

% Suppress page numbers
\pagestyle{empty} 

% Hyperlinks
\usepackage{hyperref}

% Small bullet
\usepackage{graphicx}

\renewcommand{\arraystretch}{1.3}
\newcolumntype{L}{>{\arraybackslash}m{12cm}}
\newcommand\indep{\protect\mathpalette{\protect\indeP}{\perp}}
\def\indeP#1#2{\mathrel{\rlap{$#1#2$}\mkern2mu{#1#2}}}

\pagestyle{headings}
\newcommand{\boxwidth}{430pt}
\theoremstyle{definition}
\newtheorem{problem}{Problem}

\newtheoremstyle{hSol}
  {1.0pt}% Space above
  {1.0pt}% Space below
  {}% bodyfont
  {}% indent
  {\bfseries}% thm head font
  {.}% punctuation after thm head
  { }% Space after thm head
  {}% thm head spec

\theoremstyle{hSol}
\newtheorem*{solution}{Solution}


%-------------------------------------------------------------------------------
% Code listing
%-------------------------------------------------------------------------------

\usepackage{listings}
\usepackage{color}

\definecolor{mygreen}{rgb}{0,0.6,0}
\definecolor{mygray}{rgb}{0.5,0.5,0.5}
\definecolor{mymauve}{rgb}{0.58,0,0.82}

\pagestyle{headings}
\lstset{tabsize=4,
  frame=shadowbox,
  showstringspaces=false,
  stringstyle=\ttfamily,
  keepspaces=true,
  breakindent=22pt,
  numbers=left,
  stepnumber=1,
  numberstyle=\tiny,
  basicstyle={\footnotesize\ttfamily},
  showspaces=false,
  flexiblecolumns=true,
  breaklines=true,
  breakautoindent=true,
  breakindent=4em,
}


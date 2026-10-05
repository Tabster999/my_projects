%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main.cls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Class for my master thesis
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASS OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}
\LoadClass[a4paper,english,12pt,twoside]{report} % Base class

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PACKAGES REQUIRED FOR CLASS LAYOUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[utf8]{inputenc}
\RequirePackage{nomencl}
\RequirePackage{caption}
\RequirePackage{fancyhdr}
\RequirePackage{listings} 
\RequirePackage{hyperref}
\RequirePackage{subfigure}
\RequirePackage{makeidx}
\RequirePackage{float}
\RequirePackage{a4wide}
\RequirePackage[dvipsnames]{xcolor}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASS OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\nomitemsep}{-\parsep}
\setlength{\nomlabelwidth}{.20\hsize}

% Captiondefinition
\captionsetup{font=small,labelfont=bf}% evtl. eins kleiner schalten

\renewcommand{\baselinestretch}{1.5}

\setlength{\headheight}{15pt}
\pagestyle{fancyplain} \fancyhf{}
\fancyhead[EL]{\bfseries\thepage}\fancyhead[ER]{\bfseries\leftmark}
\fancyhead[OR]{\bfseries\thepage}\fancyhead[OL]{\bfseries\rightmark}
\cfoot{}


\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}} \renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\plainheadrulewidth}{0.4pt}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\plainheadrulewidth}{0.4pt}
\renewcommand{\baselinestretch}{1.2}

\hfuzz=0.1pt \vfuzz=0.1pt \hbadness=1000 \vbadness=1000 \tolerance=8000

\setlength{\voffset}{-0.5cm}
\setlength{\hoffset}{0cm}
\setlength{\textheight}{23.4cm}
\setlength{\textwidth}{16cm}
\setlength{\footskip}{1.1cm}
\setlength{\headsep}{1cm}



\setlength{\subfigcapskip}{-10pt} \setlength{\subfigtopskip}{-10pt} \setcounter{tocdepth}{2}

\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.05}

%fügt unter dem Befehl \RN{"Zahl"} die entsprechende Zahl in römischer Schreibweise ein
\newcommand{\RN}[1]{\uppercase\expandafter{\romannumeral#1}}

\setlength{\parindent}{1.5em}

\nonfrenchspacing

\hyphenation{Ladungs-trager-dichte Beispiel-wort}

\makeindex

\lstset{numbers=left, numberstyle=\tiny, numbersep=5pt}
\lstset{language=Matlab}
\lstset{keywordstyle=\color{blue}}
\lstset{commentstyle=\color{ForestGreen}}
\hypersetup{hypertexnames=true}

\setlength\parindent{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILTER IRRELEVANT WARNINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{silence}

% The page header is deliberately disabled in my-resume, so the warning by scrlayer-scrpage is invalid. 
% \WarningFilter{fancyhdr}{\headheight is too small (12.0pt)}


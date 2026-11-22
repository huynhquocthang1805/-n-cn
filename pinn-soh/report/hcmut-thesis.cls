% =====================================================================
%  hcmut-thesis.cls -- HCMUT report class (Specialized Project template)
%  Based on https://github.com/hoang-himself/hcmut-thesis (LPPL 1.3c)
%
%  Class options:
%    english     (default) English labels on cover page and headings
%    vietnamese  Vietnamese labels (Mục lục, Chương, Tài liệu tham khảo...)
%    draft/final and any other option is passed on to memoir
% =====================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-thesis}[2026/11/09 HCMUT Specialized Project template]

\newif\if@vietnamese
\@vietnamesefalse
\DeclareOption{vietnamese}{\@vietnamesetrue}
\DeclareOption{english}{\@vietnamesefalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper,oneside]{memoir}

\RequirePackage[utf8]{vntex,inputenc}
\RequirePackage{anyfontsize,mathptmx}
\RequirePackage[T5]{fontenc}

\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{tikz}
\RequirePackage[center]{caption}

\RequirePackage{fancyhdr,hyperref}
\RequirePackage[super,square,sort]{natbib}

\if@vietnamese
  \RequirePackage[vietnamese]{babel}
\else
  \RequirePackage[english]{babel}
\fi

% ---------------------------------------------------------------------
%  Language-dependent labels
% ---------------------------------------------------------------------
\newcommand{\MonthEN}{%
  \ifcase \the\month
  \or January\or February\or March\or April\or May\or June
  \or July\or August\or September\or October\or November\or December
  \fi}

\if@vietnamese
  \newcommand{\@upperuniname}{Đại học Quốc gia Thành phố Hồ Chí Minh}
  \newcommand{\@uniname}{Trường Đại học Bách khoa}
  \newcommand{\@deptname}{Khoa Khoa học và Kỹ thuật Máy tính}
  \newcommand{\@reporttype}{Đồ án chuyên ngành}
  \newcommand{\@stumajor}{Khoa học Máy tính}
  \newcommand{\@reportplace}{TP. Hồ Chí Minh}
  \newcommand{\@reporttime}{tháng \the\month/\the\year}
  \newcommand{\@majorlabel}{Ngành}
  \newcommand{\@supervisorlabel}{Giảng viên hướng dẫn}
  \newcommand{\@studentlabel}{Sinh viên thực hiện}
  \newcommand{\@committeelabel}{Hội đồng}
  \newcommand{\@reviewerlabel}{Giảng viên phản biện}
  \newcommand{\@bibtitle}{Tài liệu tham khảo}
  \addto\captionsvietnamese{%
    \renewcommand{\contentsname}{Mục lục}%
    \renewcommand{\listtablename}{Danh sách bảng}%
    \renewcommand{\listfigurename}{Danh sách hình vẽ}%
    \renewcommand{\tablename}{Bảng}%
    \renewcommand{\figurename}{Hình}%
    \renewcommand{\chaptername}{Chương}%
    \renewcommand{\appendixname}{Phụ lục}%
    \renewcommand{\bibname}{Tài liệu tham khảo}%
  }
  % Vietnamese names for \cref / \Cref (if cleveref is loaded in main.tex)
  \AtBeginDocument{%
    \@ifpackageloaded{cleveref}{%
      \crefname{chapter}{chương}{chương}\Crefname{chapter}{Chương}{Chương}%
      \crefname{section}{mục}{mục}\Crefname{section}{Mục}{Mục}%
      \crefname{subsection}{mục}{mục}\Crefname{subsection}{Mục}{Mục}%
      \crefname{appendix}{phụ lục}{phụ lục}\Crefname{appendix}{Phụ lục}{Phụ lục}%
      \crefname{figure}{hình}{hình}\Crefname{figure}{Hình}{Hình}%
      \crefname{table}{bảng}{bảng}\Crefname{table}{Bảng}{Bảng}%
      \crefname{equation}{phương trình}{phương trình}%
      \Crefname{equation}{Phương trình}{Phương trình}%
      \crefname{enumi}{mục}{mục}\Crefname{enumi}{Mục}{Mục}%
      \crefname{footnote}{chú thích}{chú thích}%
      \Crefname{footnote}{Chú thích}{Chú thích}%
      \def\crefpairconjunction{ và }%
      \def\crefmiddleconjunction{, }%
      \def\creflastconjunction{ và }%
      \def\crefrangeconjunction{ đến }%
    }{}%
  }
\else
  \newcommand{\@upperuniname}{Vietnam National University Ho Chi Minh City}
  \newcommand{\@uniname}{Ho Chi Minh City University of Technology}
  \newcommand{\@deptname}{Faculty of Computer Science and Engineering}
  \newcommand{\@reporttype}{Specialized Project}
  \newcommand{\@stumajor}{Computer Science}
  \newcommand{\@reportplace}{Ho Chi Minh City}
  \newcommand{\@reporttime}{\MonthEN\ \the\year}
  \newcommand{\@majorlabel}{Major}
  \newcommand{\@supervisorlabel}{Supervisor(s)}
  \newcommand{\@studentlabel}{Students}
  \newcommand{\@committeelabel}{Council}
  \newcommand{\@reviewerlabel}{Reviewer}
  \newcommand{\@bibtitle}{References}
\fi

% ---------------------------------------------------------------------
%  Setters used in main.tex
% ---------------------------------------------------------------------
\newcommand{\upperuniname}[1]{\renewcommand{\@upperuniname}{#1}}
\newcommand{\uniname}[1]{\renewcommand{\@uniname}{#1}}
\newcommand{\deptname}[1]{\renewcommand{\@deptname}{#1}}
\newcommand{\reporttype}[1]{\renewcommand{\@reporttype}{#1}}
\newcommand{\stumajor}[1]{\renewcommand{\@stumajor}{#1}}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

\renewcommand{\@title}{Report title}

\newcommand{\@supervisor}{Supervisor}
\newcommand{\supervisor}[1]{\renewcommand{\@supervisor}{#1}}

% Optional fields: shown on the cover only when set
\newcommand{\@defcouncil}{}
\newcommand{\defcouncil}[1]{\renewcommand{\@defcouncil}{#1}}
\newcommand{\@reviewer}{}
\newcommand{\reviewer}[1]{\renewcommand{\@reviewer}{#1}}

\newcommand{\@stuname}{Student name -- xxxxxxx}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}

% ---------------------------------------------------------------------
%  Page layout
% ---------------------------------------------------------------------
\usetikzlibrary{calc}

\geometry{
  paper=a4paper,
  left=3cm,
  right=2cm,
  vmargin=2cm,
  includeheadfoot=true,
  headheight=30pt
}

\renewcommand{\large}{\fontsize{15}{15}\selectfont}
\renewcommand{\Large}{\fontsize{20}{20}\selectfont}

\OnehalfSpacing
\raggedbottom
\renewcommand{\arraystretch}{1.2}

% Numbering / table-of-contents depth
\setsecnumdepth{subsection}
\settocdepth{subsection}

\setlength{\cfttablenumwidth}{3em}
\setlength{\cftfigurenumwidth}{3em}

\AtBeginDocument{\renewcommand*\bibname{\@bibtitle}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

% Header / footer
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\bfseries\nouppercase\rightmark}
\fancyfoot[C]{\thepage}

% ---------------------------------------------------------------------
%  Cover page
% ---------------------------------------------------------------------
\def\coverpage{
  \pagestyle{empty}
  \large\bfseries

  \begin{tikzpicture}[remember picture,overlay]
    \draw[line width=3pt](
    $(current page.north west) + (3.0cm,-2.0cm)$
    ) rectangle (
    $(current page.south east) + (-2.0cm,2.0cm)$
    );
    \draw[line width=1pt](
    $(current page.north west) + (2.9cm,-1.9cm)$
    ) rectangle (
    $(current page.south east) + (-1.9cm,1.9cm)$
    );
  \end{tikzpicture}

  \vspace*{-3\baselineskip}
  \begin{center}
    \MakeUppercase{\@upperuniname}

    \MakeUppercase{\@uniname}

    \MakeUppercase{\@deptname}
    \vfill
    \includegraphics[scale=0.4]{graphics/hcmut.png}
    \vfill
    \MakeUppercase{\@reporttype}
    \vfill
    {\Large\MakeUppercase{\parbox[c]{0.8\linewidth}{\centering\@title}}}
    \vfill
    {\mdseries \@majorlabel: \@stumajor}
  \end{center}
  \vfill
  \hfill\parbox[t]{13cm}{
    \ifx\@defcouncil\@empty\else
      \MakeUppercase{\@committeelabel}: \@defcouncil\par
    \fi
    \MakeUppercase{\@supervisorlabel}:\par
    {\mdseries\@supervisor\par}
    \ifx\@reviewer\@empty\else
      \MakeUppercase{\@reviewerlabel}:\par
      {\mdseries\@reviewer\par}
    \fi

    \hspace*{3cm} ---o0o---\par

    \MakeUppercase{\@studentlabel}:\par
    {\mdseries\@stuname}
  }
  \vfill
  \mdseries
  \begin{center}
    \MakeUppercase{\@reportplace, \@reporttime}
  \end{center}
  \normalsize
  \cleardoublepage

  \pagestyle{fancy}
}

% ---------------------------------------------------------------------
%  Unnumbered front-matter pages (Declaration, Acknowledgements, ...)
% ---------------------------------------------------------------------
\providecommand{\phantomsection}{}
\newenvironment{preface}[1]{
  \vspace*{\stretch{2}}
  \thispagestyle{plain}
  {\noindent \bfseries \Huge #1}%
  \phantomsection\addcontentsline{toc}{chapter}{#1}
  \markboth{#1}{#1}
  \begin{center}
  \end{center}
}{
  \vspace*{\stretch{5}}
  \clearpage
}

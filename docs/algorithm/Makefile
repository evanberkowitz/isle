#
# Build all LaTeX PDF documents in the current folder.
#

LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all
all: hubbardFermiAction.pdf

%.pdf : %.tex
	$(LATEX) $(basename $@)
	$(BIBTEX) $(basename $@)
	$(LATEX) $(basename $@)
	$(LATEX) $(basename $@)

.PHONY: clean
clean:
	rm -f *.log
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.out
	rm -f *.toc

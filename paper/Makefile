CC:=latexmk
OPTIONS:=-pdf
TARGET:=main

all: onimage figures
	$(CC) $(OPTIONS) $(TARGET)

onimage:
	pdflatex onimage.dtx

figures: overview visualization

overview:
	$(CC) $(OPTIONS) figures/concept/overview.tex

visualization:
	$(CC) $(OPTIONS) figures/concept/visualization.tex

clean_partial:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.dvi
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.glo
	rm -f *.hd
	rm -f *.idx
	rm -f *.ins
	rm -f *.log
	rm -f *.out
	rm -f *main.toc
	rm -f archive.zip

clean: clean_partial
	rm -f *.pdf

fresh: clean all

zip: fresh
	$(MAKE) clean_partial
	zip archive.zip -r *

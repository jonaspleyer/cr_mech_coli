CC:=latexmk
OPTIONS:=-pdf
TARGET:=main

all: onimage
	$(CC) $(OPTIONS) $(TARGET)

onimage:
	pdflatex onimage.dtx

clean_partial:
	rm -f $(TARGET).aux
	rm -f $(TARGET).bbl
	rm -f $(TARGET).blg
	rm -f $(TARGET).dvi
	rm -f $(TARGET).fdb_latexmk
	rm -f $(TARGET).log
	rm -f $(TARGET).out
	rm -f $(TARGET).fls

	# Onimage
	rm -f onimage.aux
	rm -f onimage.glo
	rm -f onimage.hd
	rm -f onimage.idx
	rm -f onimage.ins
	rm -f onimage.log
	rm -f onimage.out
	rm -f onimage.pdf

clean: clean_partial
	rm -f $(TARGET).pdf

fresh: clean all

zip: fresh
	$(MAKE) clean_partial
	zip archive.zip -r *

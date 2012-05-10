main.pdf: main.tex refs.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm main.blg main.bbl main.log main.pdf main.aux


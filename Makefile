.PHONY: all data analysis sqlite sqlite-full app clean

all: data analysis

data:
	python src/prepare_data.py

analysis:
	python src/analysis.py

# slim DB committed to the repo and used by the hosted app
sqlite:
	python -m src.build_sqlite --slim

# full local-only DB with the raw observation tables (~960 MB, gitignored)
sqlite-full:
	python -m src.build_sqlite

app:
	streamlit run app.py

clean:
	rm -f outputs/*.parquet outputs/esg.db outputs/esg_full.db

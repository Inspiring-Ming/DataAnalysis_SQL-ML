.PHONY: all data analysis app clean

all: data analysis

data:
	python src/prepare_data.py

analysis:
	python src/analysis.py

app:
	streamlit run app.py

clean:
	rm -f outputs/*.parquet

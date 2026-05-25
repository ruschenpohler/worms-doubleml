.PHONY: all validate prepare eda ate cate blp robustness figures clean

all: validate prepare eda ate cate blp robustness figures

validate:
	uv run python src/00_validate.py

prepare: validate
	uv run python src/01_prepare.py

eda: prepare
	uv run python -m src.eda

ate: prepare
	uv run python src/02_ate_doubleml.py

cate: prepare
	uv run python src/03_cate_forest.py

blp: prepare
	uv run python src/04_blp.py

robustness: blp
	uv run python src/05_robustness.py

figures: blp robustness
	uv run python src/06_figures.py

clean:
	rm -rf data/processed/ results/ figures/ cas/store/ cas/shadow/

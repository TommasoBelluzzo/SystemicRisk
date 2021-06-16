SPHINX_EXE  = python -m sphinx
SOURCE_DIR  = _source
BUILD_DIR   = _build
SPHINX_OPTS = -n

help:
	@$(SPHINX_EXE) --help

.PHONY: help Makefile

%: Makefile
	@rm -rf "$(BUILD_DIR)/doctest/*"
	@rm -rf "$(BUILD_DIR)/linkcheck/*"
	@rm -rf "$(BUILD_DIR)/coverage/*"
	@rm -rf "$(BUILD_DIR)/$@/*"
	@$(SPHINX_EXE) -b doctest -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/doctest" -W $(O)
	@$(SPHINX_EXE) -b linkcheck -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/linkcheck" -W $(O)
	@$(SPHINX_EXE) -b coverage -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/coverage" -W $(O)
	@$(SPHINX_EXE) -b $@ -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/$@" $(SPHINX_OPTS) $(O)

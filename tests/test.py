import os
from pathlib import Path
import shutil
import tempfile
import unittest


from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


SAVE = os.getenv('SAVE_NOTEBOOKS') == '1'
OUT_DIR = Path('docs/_executed')




class NotebookTestCase(unittest.TestCase):
    """
    Base test case for running tutorial notebooks.
    """

    NOTEBOOK: Path | None = None
    TIMEOUT = 300


    def setUp(self) -> None:
        """
        Run notebooks in a temporary directory.
        """
        
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)


    def tearDown(self) -> None:
        """
        Delete the temporary directory contents.
        """
        
        self.tmp.cleanup()


    def run_notebook(self) -> None:
        """
        Run and then save the notebook.
        """
        
        with self.NOTEBOOK.open() as file:
            nb = nbformat.read(file, as_version=4)
        
        ep = ExecutePreprocessor(
            timeout=self.TIMEOUT,
            kernel_name='python3',
            allow_errors=False,
            )
        
        ep.preprocess(nb, {'metadata': {'path': str(self.workdir)}})
        
        # save notebook
        if SAVE:
            out = OUT_DIR / self.NOTEBOOK.name
            if not out.parent.is_dir():
                out.parent.mkdir(parents=True)
            with out.open('w', encoding='utf-8') as file:
                nbformat.write(nb, file)


    def test_notebook_runs(self) -> None:
        """
        Notebook run test.
        """
        
        if self.NOTEBOOK is None:
            self.skipTest('abstract base class')
        
        self.run_notebook()




class TestCalibrationErrorPropagation(NotebookTestCase):
    """
    Run the calibration error propagation notebook.
    """

    NOTEBOOK = Path('docs/tests/calibration_error_propagation.ipynb')




class TestSExtractorComparison(NotebookTestCase):
    """
    Run the calibration error propagation notebook.
    """

    NOTEBOOK = Path('docs/tests/sextractor_comparison.ipynb')


    def setUp(self) -> None:
        """
        Run notebooks in a temporary directory.
        """
        
        super().setUp()
        
        # copy input files to temp dir
        shutil.copytree('docs/tests/sextractor_comparison', self.workdir / 'sextractor_comparison')




class TestCalibrations(NotebookTestCase):
    """
    Run the applying corrections tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/applying_corrections.ipynb')




class TestBackgrounds(NotebookTestCase):
    """
    Run the backgrounds tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/backgrounds.ipynb')




class TestFinders(NotebookTestCase):
    """
    Run the source finder tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/finders.ipynb')




class TestInstruments(NotebookTestCase):
    """
    Run the instruments tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/instruments.ipynb')




class TestLocalBackgrounds(NotebookTestCase):
    """
    Run the local background tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/local_backgrounds.ipynb')




class TestReduction(NotebookTestCase):
    """
    Run the reduction tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/reduction.ipynb')




class TestTimingMethods(NotebookTestCase):
    """
    Run the timing methods tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/timing_methods.ipynb')




class TestVisualisation(NotebookTestCase):
    """
    Run the visualisation tutorial.
    """
    
    NOTEBOOK = Path('docs/tutorials/visualisation.ipynb')







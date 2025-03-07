import os
import warnings
from abc import ABC, abstractmethod
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import cupy as cp
from typing import List


import useful_rdkit_utils as uru

try:
    from openeye import oechem
    from openeye import oeomega
    from openeye import oeshape
    from openeye import oedocking
    import joblib
except ImportError:
    # Since openeye is a commercial software package, just pass with a warning if not available
    warnings.warn(f"Openeye packages not available in this environment; do not attempt to use ROCSEvaluator or "
                  f"FredEvaluator")
from rdkit import Chem, DataStructs
import pandas as pd
from sqlitedict import SqliteDict

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, mol):
        pass

    @property
    @abstractmethod
    def counter(self):
        pass


class MWEvaluator(Evaluator):
    """A simple evaluation class that calculates molecular weight, this was just a development tool
    """

    def __init__(self):
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        return uru.MolWt(mol)


class FPEvaluator(Evaluator):
    """An evaluator class that calculates a fingerprint Tanimoto to a reference molecule
    """

    def __init__(self, input_dict):
        self.ref_smiles = input_dict["query_smiles"]
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()
        self.ref_mol = Chem.MolFromSmiles(self.ref_smiles)
        self.ref_fp = self.fpgen.GetFingerprint(self.ref_mol)
        self.num_evaluations = 0
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, rd_mol_in):
        self.num_evaluations += 1
        rd_mol_fp = self.fpgen.GetFingerprint(rd_mol_in)
        return DataStructs.TanimotoSimilarity(self.ref_fp, rd_mol_fp)


class GPUFPEvaluator(Evaluator):
    """An evaluator class that calculates fingerprint Tanimoto similarities using GPU acceleration
    """
    def __init__(self, input_dict: dict, batch_size: int = 1000):
        """Initialize the GPU-accelerated fingerprint evaluator
        
        Args:
            input_dict: Dictionary containing query_smiles
            batch_size: Size of batches for GPU processing
        """
        self.ref_smiles = input_dict["query_smiles"]
        self.batch_size = batch_size
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()
        self.ref_mol = Chem.MolFromSmiles(self.ref_smiles)
        
        # Generate reference fingerprint and convert to GPU array
        ref_fp = self.fpgen.GetFingerprint(self.ref_mol)
        ref_fp_array = np.zeros((1, ref_fp.GetNumBits()), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(ref_fp, ref_fp_array[0])
        self.ref_fp_gpu = cp.asarray(ref_fp_array)
        
        self.num_evaluations = 0
        self._cached_fps = {}  # Cache for fingerprints
        
    @property
    def counter(self):
        return self.num_evaluations
    
    def _fp_to_array(self, fp) -> np.ndarray:
        """Convert RDKit fingerprint to numpy array"""
        arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    def _calculate_tanimoto_gpu(self, fp_array: np.ndarray) -> float:
        """Calculate Tanimoto similarity using GPU
        
        Args:
            fp_array: Numpy array of fingerprint bits
            
        Returns:
            Tanimoto similarity score
        """
        # Convert to GPU array
        fp_gpu = cp.asarray(fp_array.reshape(1, -1))
        
        # Calculate Tanimoto similarity on GPU
        intersection = cp.dot(self.ref_fp_gpu, fp_gpu.T)
        sum_fps = cp.sum(self.ref_fp_gpu, axis=1)[:, None] + cp.sum(fp_gpu, axis=1)
        union = sum_fps - intersection
        
        # Return similarity score
        return float(cp.asnumpy(intersection / union)[0, 0])
    
    def evaluate(self, rd_mol_in: Chem.Mol) -> float:
        """Evaluate Tanimoto similarity between input molecule and reference
        
        Args:
            rd_mol_in: Input RDKit molecule
            
        Returns:
            Tanimoto similarity score
        """
        self.num_evaluations += 1
        
        # Generate canonical SMILES for caching
        smi = Chem.MolToSmiles(rd_mol_in, canonical=True)
        
        # Check cache first
        if smi in self._cached_fps:
            return self._cached_fps[smi]
        
        # Generate fingerprint and convert to array
        mol_fp = self.fpgen.GetFingerprint(rd_mol_in)
        fp_array = self._fp_to_array(mol_fp)
        
        # Calculate similarity
        score = self._calculate_tanimoto_gpu(fp_array)
        
        # Cache the result
        self._cached_fps[smi] = score
        
        return score
    
    def evaluate_batch(self, mols: List[Chem.Mol]) -> np.ndarray:
        """Evaluate multiple molecules in batch for better GPU utilization
        
        Args:
            mols: List of RDKit molecules
            
        Returns:
            Array of Tanimoto similarity scores
        """
        self.num_evaluations += len(mols)
        
        # Process in batches
        all_scores = []
        for i in range(0, len(mols), self.batch_size):
            batch = mols[i:i + self.batch_size]
            
            # Generate fingerprints for batch
            fps = [self.fpgen.GetFingerprint(mol) for mol in batch]
            
            # Convert to array
            fp_array = np.vstack([self._fp_to_array(fp) for fp in fps])
            
            # Convert to GPU array
            fp_gpu = cp.asarray(fp_array)
            
            # Calculate similarities
            intersection = cp.dot(self.ref_fp_gpu, fp_gpu.T)
            sum_fps = cp.sum(self.ref_fp_gpu, axis=1)[:, None] + cp.sum(fp_gpu, axis=1)
            union = sum_fps - intersection
            scores = cp.asnumpy(intersection / union)[0]
            
            all_scores.extend(scores)
        
        return np.array(all_scores)


class ROCSEvaluator(Evaluator):
    """An evaluator class that calculates a ROCS score to a reference molecule
    """

    def __init__(self, input_dict):
        ref_filename = input_dict['query_molfile']
        ref_fs = oechem.oemolistream(ref_filename)
        self.ref_mol = oechem.OEMol()
        oechem.OEReadMolecule(ref_fs, self.ref_mol)
        self.max_confs = 50
        self.score_cache = {}
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def set_max_confs(self, max_confs):
        """Set the maximum number of conformers generated by Omega
        :param max_confs:
        """
        self.max_confs = max_confs

    def evaluate(self, rd_mol_in):
        """Generate conformers with Omega and evaluate the ROCS overlay of conformers to a reference molecule
        :param rd_mol_in: Input RDKit molecule
        :return: ROCS Tanimoto Combo score, returns -1 if conformer generation fails
        """
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(rd_mol_in)
        # Look up to see if we already processed this molecule
        arc_tc = self.score_cache.get(smi)
        if arc_tc is not None:
            tc = arc_tc
        else:
            fit_mol = oechem.OEMol()
            oechem.OEParseSmiles(fit_mol, smi)
            ret_code = generate_confs(fit_mol, self.max_confs)
            if ret_code:
                tc = self.overlay(fit_mol)
            else:
                tc = -1.0
            self.score_cache[smi] = tc
        return tc

    def overlay(self, fit_mol):
        """Use ROCS to overlay two molecules
        :param fit_mol: OEMolecule
        :return: Combo Tanimoto for the overlay
        """
        prep = oeshape.OEOverlapPrep()
        prep.Prep(self.ref_mol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(self.ref_mol)
        prep.Prep(fit_mol)
        score = oeshape.OEBestOverlayScore()
        overlay.BestOverlay(score, fit_mol, oeshape.OEHighestTanimoto())
        return score.GetTanimotoCombo()


class LookupEvaluator(Evaluator):
    """A simple evaluation class that looks up values from a file.
    This is primarily used for testing.
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        ref_filename = input_dictionary['ref_filename']
        ref_df = pd.read_csv(ref_filename)
        self.ref_dict = dict([(a, b) for a, b in ref_df[['SMILES', 'val']].values])

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        return self.ref_dict[smi]

class DBEvaluator(Evaluator):
    """A simple evaluator class that looks up values from a database.
    This is primarily used for benchmarking
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        self.db_prefix = input_dictionary['db_prefix']
        db_filename = input_dictionary['db_filename']
        self.ref_dict = SqliteDict(db_filename)

    def __repr__(self):
        return "DBEvalutor"


    @property
    def counter(self):
        return self.num_evaluations


    def evaluate(self, smiles):
        self.num_evaluations += 1
        res = self.ref_dict.get(f"{self.db_prefix}{smiles}")
        if res is None:
            return np.nan
        else:
            if res == -500:
                return np.nan
            return res
    

class FredEvaluator(Evaluator):
    """An evaluator class that docks a molecule with the OEDocking Toolkit and returns the score
    """

    def __init__(self, input_dict):
        du_file = input_dict["design_unit_file"]
        if not os.path.isfile(du_file):
            raise FileNotFoundError(f"{du_file} was not found or is a directory")
        self.dock = read_design_unit(du_file)
        self.num_evaluations = 0
        self.max_confs = 50

    @property
    def counter(self):
        return self.num_evaluations

    def set_max_confs(self, max_confs):
        """Set the maximum number of conformers generated by Omega
        :param max_confs:
        """
        self.max_confs = max_confs

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        mc_mol = oechem.OEMol()
        oechem.OEParseSmiles(mc_mol, smi)
        confs_ok = generate_confs(mc_mol, self.max_confs)
        score = 1000.0
        docked_mol = oechem.OEGraphMol()
        if confs_ok:
            ret_code = self.dock.DockMultiConformerMolecule(docked_mol, mc_mol)
        else:
            ret_code = oedocking.OEDockingReturnCode_ConformerGenError
        if ret_code == oedocking.OEDockingReturnCode_Success:
            dock_opts = oedocking.OEDockOptions()
            sd_tag = oedocking.OEDockMethodGetName(dock_opts.GetScoreMethod())
            # this is a stupid hack, I need to figure out how to do this correctly
            oedocking.OESetSDScore(docked_mol, self.dock, sd_tag)
            score = float(oechem.OEGetSDData(docked_mol, sd_tag))
        return score


def generate_confs(mol, max_confs):
    """Generate conformers with Omega
    :param max_confs: maximum number of conformers to generate
    :param mol: input OEMolecule
    :return: Boolean Omega return code indicating success of conformer generation
    """
    rms = 0.5
    strict_stereo = False
    omega = oeomega.OEOmega()
    omega.SetRMSThreshold(rms)  # Word to the wise: skipping this step can lead to significantly different charges!
    omega.SetStrictStereo(strict_stereo)
    omega.SetMaxConfs(max_confs)
    error_level = oechem.OEThrow.GetLevel()
    # Turn off OEChem warnings
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
    status = omega(mol)
    # Turn OEChem warnings back on
    oechem.OEThrow.SetLevel(error_level)
    return status


def read_design_unit(filename):
    """Read an OpenEye design unit
    :param filename: design unit filename (.oedu)
    :return: a docking grid
    """
    du = oechem.OEDesignUnit()
    rfs = oechem.oeifstream()
    if not rfs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    du = oechem.OEDesignUnit()
    if not oechem.OEReadDesignUnit(rfs, du):
        oechem.OEThrow.Fatal("Failed to read design unit")
    if not du.HasReceptor():
        oechem.OEThrow.Fatal("Design unit %s does not contain a receptor" % du.GetTitle())
    dock_opts = oedocking.OEDockOptions()
    dock = oedocking.OEDock(dock_opts)
    dock.Initialize(du)
    return dock


def test_fred_eval():
    """Test function for the Fred docking Evaluator
    :return: None
    """
    input_dict = {"design_unit_file": "data/2zdt_receptor.oedu"}
    fred_eval = FredEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = fred_eval.evaluate(mol)
    print(score)


def test_rocs_eval():
    """Test function for the ROCS evaluator
    :return: None
    """
    input_dict = {"query_molfile": "data/2chw_lig.sdf"}
    rocs_eval = ROCSEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    combo_score = rocs_eval.evaluate(mol)
    print(combo_score)


class MLClassifierEvaluator(Evaluator):
    """An evaluator class the calculates a score based on a trained ML model
    """

    def __init__(self, input_dict):
        self.cls = joblib.load(input_dict["model_filename"])
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        fp = uru.mol2morgan_fp(mol)
        return self.cls.predict_proba([fp])[:,1][0]


def test_ml_classifier_eval():
    """Test function for the ML Classifier Evaluator
    :return: None
    """
    input_dict = {"model_filename": "mapk1_modl.pkl"}
    ml_cls_eval = MLClassifierEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = ml_cls_eval.evaluate(mol)
    print(score)


if __name__ == "__main__":
    test_rocs_eval()

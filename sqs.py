import numpy as np
import random
from ase.io import read
import ase
from ase import Atoms
from time import time
import json
import warnings

class SQS:
    
    def __init__(self,max_m=3,conv_thr=0.0):
        
        if np.isscalar(conv_thr):
            self.conv_thr = np.ones(max_m) * conv_thr
        else:
            self.conv_thr = np.array(conv_thr)
            if len(self.conv_thr) != max_m:
                raise ValueError(f'conv_thr should be a scalar or list with {max_m} convergence threshold values')

        self.max_m = max_m            
            
    def read_atoms(self,filein='POSCAR',multiply=False,nx=1,ny=1,nz=1):
        
        if type(filein) == Atoms:
            self.atoms = filein.copy()
        else:
            self.atoms = read(filein)
        
        if multiply:
            self.atoms *= (nx,ny,nz)
        
        self.nspecies = len(np.unique(self.atoms.get_chemical_symbols()))
        self.natoms = len(self.atoms)
        
        self.restore_atoms = False

    def write_sqs_atoms(self,fileout='POSCAR',**kwargs):
    
        if hasattr(self, 'sqs_atoms'):
            self.sqs_atoms.write(fileout,**kwargs)
        elif hasattr(self, 'atoms'):
            warnings.warn('Did not find sqs_atoms, using original atoms instead')
            self.atoms.write(fileout,**kwargs)

    def select_sublattice(self,sublattice):
        
        if self.nspecies == 1:
            raise ValueError('There are only one species on Atoms object')
            
        chemical_symbols = np.array(self.atoms.get_chemical_symbols())
        
        if not sublattice in chemical_symbols:
            raise ValueError('The provided sublattice does not exist on Atoms object')
            
        retain_index = (chemical_symbols == sublattice).nonzero()[0]
        remove_index = (chemical_symbols != sublattice).nonzero()[0]
        
        self.full_atoms = self.atoms.copy()
        self.full_natoms = self.natoms
        self.atoms_other = self.atoms.copy()
        self.restore_atoms = True
        
        del self.atoms[remove_index]
        self.natoms = len(self.atoms)
        
        del self.atoms_other[retain_index]
    
    def restore_full_atoms(self,atoms):
        
        atoms_other = self.atoms_other
        
        new_atoms = atoms + atoms_other
        
        return new_atoms
            
    def create_neighbor_list(self, skin_dist=0.01, twoD=False, verbose=True):
        
        if verbose:
            print(f'Creating neighbor list for {self.natoms} atoms')
        
        translation_vectors = [[ 0, 0, 0],[ 1, 0, 0],[-1, 0, 0],[ 0, 1, 0],[ 0,-1, 0],[ 1, 1, 0],[-1,-1, 0],[ 1,-1, 0],[-1, 1, 0],
                               [ 0, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 0, 1, 1],[ 0,-1, 1],[ 1, 1, 1],[-1,-1, 1],[ 1,-1, 1],[-1, 1, 1],
                               [ 0, 0,-1],[ 1, 0,-1],[-1, 0,-1],[ 0, 1,-1],[ 0,-1,-1],[ 1, 1,-1],[-1,-1,-1],[ 1,-1,-1],[-1, 1,-1]]

        self.twoD = twoD
        
        if twoD:
            self.n_trans = 9
            if verbose:
                print(f'Structure is 2D: there are {self.n_trans} PBC neighbor cells')
        else:
            self.n_trans = 27
            if verbose:
                print(f'Structure is 3D: there are {self.n_trans} PBC neighbor cells')     
        
        if verbose:
            print('Computing distance matrix', end=' ... ')
        
        t0 = time()    
        distance_matrix = np.ones((self.natoms,self.natoms))*1000
        
        pos = self.atoms.get_positions()
        latt = self.atoms.get_cell()

        for ncell in range(self.n_trans):
            pbc_atoms = self.atoms.copy()
            pbc_atoms.translate(np.dot(translation_vectors[ncell],latt))
            pos_pbc = pbc_atoms.get_positions()
            for ia in range(self.natoms):
                for ja in range(ia+1,self.natoms):
                    dist = np.linalg.norm(pos[ia]-pos_pbc[ja])
                    if dist < distance_matrix[ia,ja]:
                        distance_matrix[ia,ja] = dist
                        distance_matrix[ja,ia] = dist

        if verbose:
            print(f'done in {time()-t0:.3f} sec')
        
        if verbose:
            print(f'Creating neighbor matrix up to {self.max_m} neighbor', end=' ... ')
        
        t0 = time()
            
        distance_matrix = np.round(distance_matrix,decimals=3)
        unique_distances = np.unique(distance_matrix.ravel())
        m_neighbor_distances = unique_distances[unique_distances.argsort()[:self.max_m]]

        for ia in range(self.natoms): distance_matrix[ia,ia] = 0.0

        neighbor_matrix = np.zeros((self.natoms,self.natoms),dtype=int)

        for ia in range(self.natoms):
            for ja in range(ia+1,self.natoms):
                for im in range(self.max_m):
                    if distance_matrix[ia,ja] <= m_neighbor_distances[im]+skin_dist and neighbor_matrix[ia,ja] == 0:
                        neighbor_matrix[ia,ja] = im+1
                        neighbor_matrix[ja,ia] = im+1
        
        if verbose:
            print(f'done in {time()-t0:.3f} sec')
        
        self.distance_matrix = distance_matrix
        self.m_neighbor_distances = m_neighbor_distances
        self.neighbor_matrix = neighbor_matrix
        
    def _random_corr(self,x):
        #return (2*x-1)**2
        return x**2
    
    def generate_trial_atoms(self,alloy_species='X'):
    
        trial_atoms = self.atoms.copy()
        #replace_index = np.random.choice(np.arange(self.natoms),size=self.n_minor,replace=False)
        replace_index = random.sample(range(self.natoms),k=self.n_minor)
        #S_array = np.ones(self.natoms)
        S_array = np.zeros(self.natoms)
        
        chemical_symbols = np.array(trial_atoms.get_chemical_symbols())
        for ia in replace_index:
            chemical_symbols[ia] = alloy_species
            #S_array[ia] = -1
            S_array[ia] = 1
        trial_atoms.set_chemical_symbols(chemical_symbols)
        
        return trial_atoms, S_array, replace_index
        
    def compute_corr(self,S_array):
        
        mcorr = np.zeros((self.max_m))
        
        for im in range(self.max_m):
            
            iatoms,jatoms = (self.neighbor_matrix == im+1).nonzero()            
            n_m_neighbors = len(iatoms) #/ self.natoms

            for ia,ja in zip(iatoms,jatoms):
                mcorr[im] += S_array[ia] * S_array[ja]
                
            mcorr[im] /= (n_m_neighbors)
            
        return mcorr
        
    def create_vacancy(self,atoms,species='X'):
        
        chemical_symbols = np.array(atoms.get_chemical_symbols())
        vacancy_index = np.sort((chemical_symbols == species).nonzero()[0])[::-1]
        del atoms[vacancy_index]
        
        return atoms
        
    def create_geometry(self,alloy_species='X',concentration=1.0,maxtrials=1000,vacancy=False,verbose=True):
            
        if not hasattr(self, 'neighbor_matrix'):
            if verbose:
                warnings.warn('Neighbor list not found')
                warnings.warn('For creating more than one SQS geometry creating the neighbor list beforehand will be faster')
        
            self.create_neighbor_list()
            
        if verbose:
            print('Creating SQS geometry')
            print()
        
        self.x = concentration / 100
        
        self.n_minor = int(np.rint((concentration * self.natoms) / 100))
        self.real_x = (self.n_minor / self.natoms)
        
        if verbose:
            print(f'Asked concentration (x): {concentration:>5.2f} %')
            print(f'Real concentration (x) : {self.real_x*100:>5.2f} % ({self.n_minor} alloying atoms)')
        
        corr_ref = self._random_corr(x=self.real_x)
        
        self.corr_ref = corr_ref
        
        if verbose:
            print()
            print(f'Target pair correlation: {corr_ref:.3e}')
    
        mcorr = np.zeros(self.max_m)
        ntrials = -1
        
        self.converged = False
        
        if verbose:
            conv_str = '     |     '.join(['Corr. '+str(m+1) for m in range(self.max_m)])
            delta_conv_str = '     |     '.join(['dCorr. '+str(m+1) for m in range(self.max_m)])
            print()
            print('        step   |     '+conv_str+'     |     '+delta_conv_str)
        
        while ntrials < maxtrials:
        
            ntrials += 1
            
            # generate trial geometry
            trial_atoms, S_array, replace_index = self.generate_trial_atoms(alloy_species=alloy_species)
            
            # compute geometry pair correlation
            mcorr = self.compute_corr(S_array)
            
            # check correlation convergence
            delta_corr = np.abs(mcorr - corr_ref)**2
            
            if verbose:
                mcorr_str = '  |  '.join([ f'{corr: 12.6e}' for corr in mcorr])
                delta_corr_str = '   |  '.join([ f'{corr: 12.6e}' for corr in delta_corr])
                print(f'    {ntrials:8d}   |  '+mcorr_str+'  |  '+delta_corr_str,end='')
            
            if np.all(delta_corr <= self.conv_thr):
            # accept geometry
                self.sqs_atoms = trial_atoms.copy()
                self.S_array = S_array
                self.replace_index = replace_index
                self.corr = mcorr
                self.delta_corr = delta_corr
                self.converged = True
                
                if verbose:
                    print(' <-- converged')
                    print()
                break
            # refuse geometry
            else:
                if verbose:
                    print('')

        if self.converged:
            if verbose:
                print(f'Converged within {ntrials} steps')
                print()
                print(f'Final correlation and delta with reference corr. up to {self.max_m} neighbors:')
                print(f'*** {ntrials:8d}   |  '+mcorr_str+'  |  '+delta_corr_str+' ***')     
        
            if vacancy:
                if verbose:
                    print()
                    print('Vacancy is set to TRUE')
                    print(f'Deleting {alloy_species} atoms')
                self.sqs_atoms = self.create_vacancy(self.sqs_atoms,species=alloy_species)
            
            if self.restore_atoms:
                self.sqs_atoms = self.restore_full_atoms(self.sqs_atoms)
            
            if verbose:
                print()
                print('SQS geometry created and stored at sqs_geometry')
                print()
        else:
            if verbose:
                print()
                print(f'*** SQS did not converged within {maxtrials} trials***')
                print()
                print(' Check you starting geometry or retry with larger conv_thr')
    
    def as_dict(self):
    
        state_dict = {}

        if hasattr(self, 'max_m'):
            state_dict['conv_thr'] = self.conv_thr
            state_dict['max_m'] = self.max_m
        
        if hasattr(self, 'atoms'):
            state_dict['atoms'] = self.atoms.todict()
            state_dict['natoms'] = self.natoms
            state_dict['nspecies'] = self.nspecies
            state_dict['restore_atoms'] = self.restore_atoms

        if hasattr(self, 'twoD'):
            state_dict['twoD'] = self.twoD
            state_dict['n_trans'] = self.n_trans

        if hasattr(self, 'neighbor_matrix'):
            state_dict['distance_matrix'] = self.distance_matrix
            state_dict['m_neighbor_distances'] = self.m_neighbor_distances
            state_dict['neighbor_matrix'] = self.neighbor_matrix
            
        if hasattr(self, 'x'):
            state_dict['x'] = self.x
            state_dict['real_x'] = self.real_x
            state_dict['n_minor'] = self.n_minor
            state_dict['corr_ref'] = self.corr_ref
            state_dict['converged'] = self.converged

        if hasattr(self, 'atoms_other'):
            state_dict['full_atoms'] = self.full_atoms.todict()
            state_dict['full_natoms'] = self.full_natoms
            state_dict['atoms_other'] = self.atoms_other.todict()

        if hasattr(self, 'sqs_atoms'):
            state_dict['sqs_atoms'] = self.sqs_atoms.todict()
            state_dict['S_array'] = self.S_array
            state_dict['replace_index'] = self.replace_index
            state_dict['corr'] = self.corr
            state_dict['delta_corr'] = self.delta_corr
            
        return state_dict
    
    def save_state(self,filename='sqs_state.json'):

        state_dict = self.as_dict()
            
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        with open(f"{filename.replace('.json','')}.json", 'w') as fp:
            json.dump(state_dict, fp,cls=NumpyEncoder)
            
    def load_state(self,filename='sqs_state.json'):
              
        with open(filename, 'r') as fp:
            state_dict = json.load(fp)
        
        if 'max_m' in state_dict:
            self.conv_thr = np.asarray(state_dict['conv_thr'])
            self.max_m = state_dict['max_m']
        
        if 'atoms' in state_dict:
            self.atoms = Atoms.fromdict(state_dict['atoms'])
            self.natoms = state_dict['natoms']
            self.nspecies = state_dict['nspecies']
            self.restore_atoms = state_dict['restore_atoms']
        
        if 'atoms_other' in state_dict:
            self.full_atoms = Atoms.fromdict(state_dict['full_atoms'])
            self.full_natoms = state_dict['full_natoms']
            self.atoms_other = Atoms.fromdict(state_dict['atoms_other'])
        
        if 'twoD' in state_dict:
            self.twoD = state_dict['twoD']
            self.n_trans = state_dict['n_trans']
        
        if 'neighbor_matrix' in state_dict:
            self.distance_matrix = np.asarray(state_dict['distance_matrix'])
            self.m_neighbor_distances = np.asarray(state_dict['m_neighbor_distances'])
            self.neighbor_matrix = np.asarray(state_dict['neighbor_matrix'])
            
        if 'x' in state_dict:
            self.x = state_dict['x']
            self.real_x = state_dict['real_x']
            self.n_minor = state_dict['n_minor']
            self.corr_ref = state_dict['corr_ref']
            self.converged = state_dict['converged']
            
        if 'sqs_atoms' in state_dict:
            self.sqs_atoms = Atoms.fromdict(state_dict['sqs_atoms'])
            self.S_array = np.asarray(state_dict['S_array'])
            self.replace_index = np.asarray(state_dict['replace_index'])
            self.corr = np.asarray(state_dict['corr'])
            self.delta_corr = np.asarray(state_dict['delta_corr'])

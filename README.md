# param-medic
Param-Medic breathes new life into MS/MS database searches by optimizing search parameter settings for your data.

## Usage:

param-medic [--debug] \<spectra file\>+

## Description:

In shotgun proteomics analysis, search engines compare tandem mass spectrometry (MS/MS) spectra with theoretical spectra to make peptide-spectrum matches (PSMs). User-specified parameters are critical to search performance. In particular:

* Precursor mass tolerance defines the peptide candidates considered for each spectrum. 
* Fragment mass tolerance (or bin size) determines how close observed and theoretical fragments must be in order to be considered a match.

For each parameter, too loose a setting yields randomly high-scoring false PSMs, while too tight excludes true PSMs. 

Param-Medic finds pairs of spectra that are likely to have been generated by the same peptide and uses these pairs to infer optimal parameters for search with Comet, Tide and other search engines. If multiple files are specified, they will be processed together. 

Param-Medic may fail if two few paired spectra are discovered, or if precursor m/z values appear to be artificially manipulated. 

## Input

* \<spectra file\>+: The path to one or more files from which to parse fragmentation spectra, in .mzML or .ms2 format.  

## Output

Estimated parameter values for precursor mass tolerance (ppm) and fragment bin size (ppm and, separately, Th).

## Options

--debug: If this flag is set, verbose debug logging will be enabled.

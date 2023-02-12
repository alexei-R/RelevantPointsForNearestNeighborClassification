"dataset_size;dimensionality;label_consolidation_level;execution_time_eppstein_ms;boundary_points_eppstein;execution_time_flores_ms;boundary_points_flores"
for (($consolidation_level = 0); $consolidation_level -lt 3; $consolidation_level++) 
{
	for (($d = 3); $d -lt 12; $d += 2) 
     	{
        	& '.\NextNeighboursReduction.exe' 4898 $d $consolidation_level
    	}
}
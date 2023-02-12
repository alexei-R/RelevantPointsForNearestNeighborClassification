"dataset_size;dimensionality;label_consolidation_level;execution_time_eppstein_ms;boundary_points_eppstein;execution_time_flores_ms;boundary_points_flores"
for (($k = 1); $k -lt 6; $k++)
{
	for (($d = 3); $d -lt 12; $d += 2) 
      {
       	& '.\NextNeighboursReduction.exe' ($k * 980) $d 2
      }
}

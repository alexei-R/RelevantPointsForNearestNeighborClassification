for (($consolidate = 0); $consolidate -lt 2; $consolidate++) 
{
    for (($k = 1); $k -lt 6; $k++)
    {
        for (($d = 3); $d -lt 12; $d += 2) 
        {
            & '.\NextNeighboursReduction.exe' ($k * 980) $d $consolidate
        }
    }
}
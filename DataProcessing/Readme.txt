Put all full length clips inside folder "Full_Clips"
Put all segments (cut clips) into folder "Cut_Clips"
Put all MFCC pngs into folder "ALL_MFCCs"

If folders "Rejected_MFCCS" or "Rejected_noise" do not exist, create them

Run first cell
-This will do the mean comparison and then store all rejects into "Rejected_noise"
Once done, run second cell
-This will find the corresponding MFCCs from "All_MFCCs" and store them in "Rejected_MFCCS"

Recommended to do a species at a time (etc Andropusdus and all its segments, run script, move results and then put next species in)

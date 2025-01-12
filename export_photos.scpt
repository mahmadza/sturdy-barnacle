tell application "Photos"
    set theSel to selection
    if (count of theSel) is greater than 0 then
        set exportPath to "/Users/mamduhzabidi/Desktop/PhotosExport/"
        repeat with i from 1 to count of theSel
            set thePhoto to item i of theSel
            export thePhoto to exportPath with using originals
        end repeat
    else
        display dialog "No photos selected"
    end if
end tell
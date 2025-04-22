//YourFile.cpp (compiled into a .dll or .so file)
#include <new> //For std::nothrow
#include "BYTETracker.h"

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    //Note: The interface this linkage region needs to use C only.
    void * ByteTrackNew(float track_thresh = 0.5, float high_thresh = 0.6, float match_thresh = 0.8, uint32_t frame_rate = 30, uint32_t track_buffer = 30)
    {
        // Note: Inside the function body, I can use C++.
        return new(std::nothrow) BYTETracker(track_thresh, high_thresh, match_thresh, frame_rate, track_buffer);
    }

    //Thanks Chris.
    void ByteTrackDelete (void *ptr)
    {
        delete reinterpret_cast<BYTETracker *>(ptr);
    }

    uint32_t ByteTrackUpdate(void *ptr, const float* const box, const uint32_t nbox, float* const out, const uint32_t n_extra = 0)
    {

        // Note: A downside here is the lack of type safety.
        // You could always internally(in the C++ library) save a reference to all
        // pointers created of type MyClass and verify it is an element in that
        //structure.
        //
        // Per comments with Andre, we should avoid throwing exceptions.
        try
        {
            BYTETracker * ref = reinterpret_cast<BYTETracker *>(ptr);
            return ref->update(box, nbox, out, n_extra);
        }
        catch(...)
        {
           return -1; //assuming -1 is an error condition.
        }
    }

    uint32_t get_tracked_tracks_count(void *ptr) { return reinterpret_cast<BYTETracker *>(ptr)->get_tracked_tracks_count(); }
	uint32_t get_lost_tracks_count(void *ptr) { return reinterpret_cast<BYTETracker *>(ptr)->get_lost_tracks_count(); }
	uint32_t get_removed_tracks_count(void *ptr) { return reinterpret_cast<BYTETracker *>(ptr)->get_removed_tracks_count(); }

	void clear_buffer(void *ptr) { return reinterpret_cast<BYTETracker *>(ptr)->clear_buffer(); }
	void clear_lost(void *ptr) { return reinterpret_cast<BYTETracker *>(ptr)->clear_lost(); }

} //End C linkage scope.

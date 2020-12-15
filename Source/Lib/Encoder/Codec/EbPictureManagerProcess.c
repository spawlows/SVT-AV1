/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <stdlib.h>
#include <string.h>

#include "EbEncHandle.h"
#include "EbUtility.h"
#include "EbPictureControlSet.h"
#include "EbSequenceControlSet.h"
#include "EbPictureManagerProcess.h"
#include "EbReferenceObject.h"
#include "EbPictureDemuxResults.h"
#include "EbPictureManagerQueue.h"
#include "EbPredictionStructure.h"
#include "EbRateControlTasks.h"
#include "EbSvtAv1ErrorCodes.h"
#include "EbEntropyCoding.h"
#include "EbLog.h"

// Token buffer is only used for palette tokens.
static INLINE unsigned int get_token_alloc(int mb_rows, int mb_cols, int sb_size_log2,
                                           const int num_planes) {
    // Calculate the maximum number of max superblocks in the image.
    const int shift          = sb_size_log2 - 4;
    const int sb_size        = 1 << sb_size_log2;
    const int sb_size_square = sb_size * sb_size;
    const int sb_rows        = ALIGN_POWER_OF_TWO(mb_rows, shift) >> shift;
    const int sb_cols        = ALIGN_POWER_OF_TWO(mb_cols, shift) >> shift;

    // One palette token for each pixel. There can be palettes on two planes.
    const int sb_palette_toks = AOMMIN(2, num_planes) * sb_size_square;

    return sb_rows * sb_cols * sb_palette_toks;
}
void alloc_palette_tokens(SequenceControlSet * scs_ptr, PictureControlSet * child_pcs_ptr);
extern MvReferenceFrame svt_get_ref_frame_type(uint8_t list, uint8_t ref_idx);
/************************************************
 * Defines
 ************************************************/
#define POC_CIRCULAR_ADD(base, offset) (((base) + (offset)))

void largest_coding_unit_dctor(EbPtr p);

/************************************************
  * Configure Picture edges
  ************************************************/
static void configure_picture_edges(SequenceControlSet *scs_ptr, PictureControlSet *ppsPtr) {
    // Tiles Initialisation
    const uint16_t pic_width_in_sb =
        (ppsPtr->parent_pcs_ptr->aligned_width + scs_ptr->sb_size_pix - 1) / scs_ptr->sb_size_pix;
    const uint16_t picture_height_in_sb =
        (ppsPtr->parent_pcs_ptr->aligned_height + scs_ptr->sb_size_pix - 1) / scs_ptr->sb_size_pix;
    unsigned x_sb_index, y_sb_index, sb_index;

    // SB-loops
    for (y_sb_index = 0; y_sb_index < picture_height_in_sb; ++y_sb_index) {
        for (x_sb_index = 0; x_sb_index < pic_width_in_sb; ++x_sb_index) {
            sb_index = (uint16_t)(x_sb_index + y_sb_index * pic_width_in_sb);
            ppsPtr->sb_ptr_array[sb_index]->picture_left_edge_flag =
                (x_sb_index == 0) ? EB_TRUE : EB_FALSE;
            ppsPtr->sb_ptr_array[sb_index]->picture_top_edge_flag =
                (y_sb_index == 0) ? EB_TRUE : EB_FALSE;
            ppsPtr->sb_ptr_array[sb_index]->picture_right_edge_flag =
                (x_sb_index == (unsigned)(pic_width_in_sb - 1)) ? EB_TRUE : EB_FALSE;
        }
    }

    return;
}
void write_stat_to_file(SequenceControlSet *scs_ptr, StatStruct stat_struct, uint64_t ref_poc);

static void picture_manager_context_dctor(EbPtr p) {
    EbThreadContext *      thread_context_ptr = (EbThreadContext *)p;
    PictureManagerContext *obj                = (PictureManagerContext *)thread_context_ptr->priv;
    EB_FREE_ARRAY(obj);
}
/************************************************
 * Picture Manager Context Constructor
 ************************************************/
EbErrorType picture_manager_context_ctor(EbThreadContext *  thread_context_ptr,
                                         const EbEncHandle *enc_handle_ptr,
                                         int                rate_control_index) {
    PictureManagerContext *context_ptr;
    EB_CALLOC_ARRAY(context_ptr, 1);
    thread_context_ptr->priv  = context_ptr;
    thread_context_ptr->dctor = picture_manager_context_dctor;

    context_ptr->picture_input_fifo_ptr =
        svt_system_resource_get_consumer_fifo(enc_handle_ptr->picture_demux_results_resource_ptr, 0);
    UNUSED(rate_control_index);
    context_ptr->picture_manager_output_fifo_ptr = svt_system_resource_get_producer_fifo(
        enc_handle_ptr->pic_mgr_res_srm, 0);
    context_ptr->picture_control_set_fifo_ptr = svt_system_resource_get_producer_fifo(
        enc_handle_ptr->picture_control_set_pool_ptr_array[0], 0); //The Child PCS Pool here

    return EB_ErrorNone;
}

void copy_buffer_info(EbPictureBufferDesc *src_ptr, EbPictureBufferDesc *dst_ptr){
    dst_ptr->width = src_ptr->width;
    dst_ptr->height = src_ptr->height;
    dst_ptr->max_width = src_ptr->max_width;
    dst_ptr->max_height = src_ptr->max_height;
    dst_ptr->stride_y = src_ptr->stride_y;
    dst_ptr->stride_cb = src_ptr->stride_cb;
    dst_ptr->stride_cr = src_ptr->stride_cr;
    dst_ptr->origin_x = src_ptr->origin_x;
    dst_ptr->origin_bot_y = src_ptr->origin_bot_y;
    dst_ptr->origin_y = src_ptr->origin_y;
    dst_ptr->stride_bit_inc_y = src_ptr->stride_bit_inc_y;
    dst_ptr->stride_bit_inc_cb = src_ptr->stride_bit_inc_cb;
    dst_ptr->stride_bit_inc_cr = src_ptr->stride_bit_inc_cr;
    dst_ptr->luma_size = src_ptr->luma_size;
    dst_ptr->chroma_size = src_ptr->chroma_size;
}

void set_tile_info(PictureParentControlSet *pcs_ptr);
/*
  walk the ref queue, and looks for a particular pic
  return NULL if not found.
*/
ReferenceQueueEntry * search_ref_in_ref_queue(
    EncodeContext *encode_context_ptr,
    uint64_t ref_poc)
{
    ReferenceQueueEntry * ref_entry_ptr = NULL;
    uint32_t ref_queue_i = encode_context_ptr->reference_picture_queue_head_index;
    // Find the Reference in the Reference Queue
    do {
        ref_entry_ptr =
            encode_context_ptr->reference_picture_queue[ref_queue_i];
        if (ref_entry_ptr->picture_number == ref_poc)
            return ref_entry_ptr;

        // Increment the reference_queue_index Iterator
        ref_queue_i = (ref_queue_i == REFERENCE_QUEUE_MAX_DEPTH - 1)
            ? 0
            : ref_queue_i + 1;

    } while (ref_queue_i != encode_context_ptr->reference_picture_queue_tail_index);

    return NULL;
}
/*
  update dependent count of any existing pic in the ref queue.
  if the picture is not in the ref queue, subesequent calls
  will attempt to do the cleaning
*/
void clean_pictures_in_ref_queue(EncodeContext *ctx)
{
    PicQueueEntry **queue = ctx->dep_cnt_picture_queue;
    ReferenceQueueEntry * ref_entry_ptr = NULL;

    //all pictures needing the clean-up are stored in dep cnt queue
    //go through all of elements, search it in the ref queue, and clean
    //it if found
    uint32_t  q_idx = ctx->dep_q_head;
    while (q_idx != ctx->dep_q_tail) {
        if (queue[q_idx]->is_done == 0){
            ref_entry_ptr = search_ref_in_ref_queue(ctx, queue[q_idx]->pic_num);
            if (ref_entry_ptr != NULL) {
                int new_dep_cnt = (int)ref_entry_ptr->dependent_count + queue[q_idx]->dep_cnt_diff;
                if (new_dep_cnt < 0)
                    SVT_ERROR(" PicMgr: Negative dep count\n");
                //printf(" search %I64i  ====Pic-MGR-Dep-Cnt-reduce found: %I64i  %i --> %i\n",
                //    queue[q_idx]->pic_num, ref_entry_ptr->picture_number, ref_entry_ptr->dependent_count, new_dep_cnt);
                ref_entry_ptr->dependent_count += queue[q_idx]->dep_cnt_diff;
                queue[q_idx]->is_done = 1;
            }
        }

        // Increment the head_index if the head is done
        if (queue[ctx->dep_q_head]->is_done)
            ctx->dep_q_head = (ctx->dep_q_head == REFERENCE_QUEUE_MAX_DEPTH - 1) ? 0 : ctx->dep_q_head + 1;

        //Increment the queue_index Iterator
        q_idx = (q_idx == REFERENCE_QUEUE_MAX_DEPTH - 1) ? 0 : q_idx + 1;
    }
}

/*
  Pic Mgr maintains a queue of pic to clear dep cnt
  if a picture comes from PD, it copies its clean-up list to the queue
*/
void copy_dep_cnt_cleaning_list(
    EncodeContext *ctx,
    PictureParentControlSet * pcs)
{
    //triggering picture hands list to PicMgr, which will do the clearing
    PicQueueEntry **queue = ctx->dep_cnt_picture_queue;

    for (uint32_t pic_i = 0; pic_i < pcs->other_updated_links_cnt; pic_i++) {

        //place pic to be cleared in queue
        //the spot should be empty
        if (queue[ctx->dep_q_tail]->is_done == 0)
            SVT_ERROR("\n pic %I64i  ERR  DepCount Queue is done\n", queue[ctx->dep_q_tail]->pic_num);

        queue[ctx->dep_q_tail]->pic_num = pcs->updated_links_arr[pic_i].pic_num;
        queue[ctx->dep_q_tail]->dep_cnt_diff = pcs->updated_links_arr[pic_i].dep_cnt_diff;
        queue[ctx->dep_q_tail]->is_done = 0;

        //advance the queue
        ctx->dep_q_tail = (ctx->dep_q_tail == REFERENCE_QUEUE_MAX_DEPTH - 1) ? 0 : ctx->dep_q_tail + 1;
    }

}
// get references used by a TPL group
// References are the source frames
static uint8_t tpl_setup_me_refs(
    SequenceControlSet              *scs_ptr,
    PictureParentControlSet         *pcs_tpl_group_frame_ptr,
    PictureParentControlSet         *pcs_tpl_base_ptr,
    uint8_t                         *ref0_count, //out
    uint8_t                         *ref1_count,
    EbBool                          *trailing_frames) {
    *ref0_count = 0;
    *ref1_count = 0;
    PredictionStructure* base_pred_struct_ptr = get_prediction_structure(
            scs_ptr->encode_context_ptr->prediction_structure_group_ptr,
            pcs_tpl_base_ptr->pred_structure,
            scs_ptr->reference_count,
            pcs_tpl_base_ptr->hierarchical_levels);
    uint32_t init_idx = base_pred_struct_ptr->init_pic_index;
    uint64_t curr_poc = pcs_tpl_group_frame_ptr->picture_number;
    uint64_t base_poc = pcs_tpl_base_ptr->picture_number;
    uint32_t tpl_base_minigop = base_pred_struct_ptr->pred_struct_period;

    uint32_t curr_minigop_entry_idx = (curr_poc > base_poc) ?
                                      (uint32_t)(curr_poc - base_poc) :
                                      (uint32_t)(curr_poc + tpl_base_minigop - base_poc);
    uint32_t pred_struct_idx = curr_minigop_entry_idx + init_idx;

    // Get the ref entry point of trailing frames, not good here
    if (*trailing_frames && pred_struct_idx + tpl_base_minigop < base_pred_struct_ptr->pred_struct_entry_count)
        pred_struct_idx += tpl_base_minigop;

    PredictionStructureEntry *frame_pred_entry = base_pred_struct_ptr->pred_struct_entry_ptr_array[pred_struct_idx];

    uint8_t ref_list_count = 0;
    for (uint8_t list_index = REF_LIST_0; list_index <= REF_LIST_1; list_index++) {
        uint8_t *ref_count_ptr = (list_index == REF_LIST_0) ? ref0_count : ref1_count;

        if (*trailing_frames) {
            ref_list_count = (list_index == REF_LIST_0) ?
                frame_pred_entry->ref_list0.reference_list_count :
                0; //Jing: why remove ref1? for poc17, it should refer to poc18 in L1
            //Jing: We can set the it as the same logic of ref_list_count_try in PD kernel
            //      But need to make sure change them at the same time.
            //      Now for simplicity, just limit it as 2 for trailing frames
            //      Corresponding logic in PD:
            //          if (pcs_ptr->enc_mode <= ENC_M6) {
            //              pcs_ptr->ref_list0_count_try = MIN(pcs_ptr->ref_list0_count, 4);
            //              pcs_ptr->ref_list1_count_try = MIN(pcs_ptr->ref_list1_count, 3);
            //          }
            //          else {
            //              pcs_ptr->ref_list0_count_try =
            //                  pcs_ptr->is_used_as_reference_flag ? MIN(pcs_ptr->ref_list0_count, 2) : MIN(pcs_ptr->ref_list0_count, 1);
            //              pcs_ptr->ref_list1_count_try =
            //                  pcs_ptr->is_used_as_reference_flag ? MIN(pcs_ptr->ref_list1_count, 2) : MIN(pcs_ptr->ref_list1_count, 1);
            //          }
            ref_list_count = MIN(ref_list_count, 2); //Jing: limit to 2 refs for trailing frames
        } else {
            ref_list_count = (list_index == REF_LIST_0) ?
                pcs_tpl_group_frame_ptr->ref_list0_count_try :
                pcs_tpl_group_frame_ptr->ref_list1_count_try;
        }

        for (uint8_t ref_idx = 0; ref_idx < ref_list_count; ref_idx++) {
            EbBool ref_in_slide_window = EB_FALSE;
            pcs_tpl_group_frame_ptr->tpl_data.ref_in_slide_window[list_index][*ref_count_ptr] = EB_FALSE;
            uint64_t ref_poc = pcs_tpl_group_frame_ptr->ref_pic_poc_array[list_index][ref_idx];
            if (*trailing_frames) {
                int delta_poc = (list_index == REF_LIST_0) ?
                    frame_pred_entry->ref_list0.reference_list[ref_idx] :
                    frame_pred_entry->ref_list1.reference_list[ref_idx];
                ref_poc = pcs_tpl_group_frame_ptr->picture_number - delta_poc;
            }

            for (uint32_t j = 0; j < pcs_tpl_base_ptr->tpl_group_size; j++) {
                if (ref_poc == pcs_tpl_base_ptr->tpl_group[j]->picture_number) {
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr] = pcs_tpl_base_ptr->tpl_group[j]->ds_pics;
                    ref_in_slide_window = EB_TRUE;
                    pcs_tpl_group_frame_ptr->tpl_data.ref_in_slide_window[list_index][*ref_count_ptr] = EB_TRUE;
                    *ref_count_ptr += 1;
                    break;
                }
            }

            if (!ref_in_slide_window) {
                if (scs_ptr->in_loop_me == 0) {
                    EbPaReferenceObject * ref_obj =
                        (EbPaReferenceObject *)pcs_tpl_group_frame_ptr->ref_pa_pic_ptr_array[list_index][*ref_count_ptr]->object_ptr;

                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].picture_number = ref_obj->picture_number;

                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].picture_ptr =
                        ref_obj->input_padded_picture_ptr;
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].sixteenth_picture_ptr = NULL;
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].quarter_picture_ptr = NULL;

                    *ref_count_ptr += 1;
                }else{
                ReferenceQueueEntry* ref_entry_ptr = search_ref_in_ref_queue(scs_ptr->encode_context_ptr, ref_poc);
                if (ref_entry_ptr && ref_entry_ptr->reference_available) {
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr] =
                        ((EbReferenceObject *)ref_entry_ptr->reference_object_ptr->object_ptr)->ds_pics;
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].picture_ptr =
                        ((EbReferenceObject *)ref_entry_ptr->reference_object_ptr->object_ptr)->input_picture;
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].sixteenth_picture_ptr =
                        ((EbReferenceObject *)ref_entry_ptr->reference_object_ptr->object_ptr)->sixteenth_input_picture;
                    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[list_index][*ref_count_ptr].quarter_picture_ptr =
                        ((EbReferenceObject *)ref_entry_ptr->reference_object_ptr->object_ptr)->quarter_input_picture;
                    *ref_count_ptr += 1;
                }
                }
            }
        }
    }

    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref0_count = *ref0_count;
    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref1_count = *ref1_count;

    // Get the temporal layer index here
    if (*trailing_frames) {
        pcs_tpl_group_frame_ptr->tpl_data.tpl_temporal_layer_index =
            frame_pred_entry->temporal_layer_index;
        pcs_tpl_group_frame_ptr->tpl_data.is_used_as_reference_flag =
            frame_pred_entry->is_referenced;

        //decode order not accurate enough when at the end of bitstream where minigop switch happens
        //and there's no way to detect the accurate dependancy/ref and decode order without caching a lot of frames,
        pcs_tpl_group_frame_ptr->tpl_data.tpl_decode_order =
            pcs_tpl_base_ptr->picture_number + frame_pred_entry->decode_order + 1;
    }
    return 0;
}
void set_tpl_controls(
    PictureParentControlSet       *pcs_ptr, uint8_t tpl_level);
static EbErrorType tpl_init_pcs_tpl_data(
    PictureParentControlSet         *pcs_tpl_group_frame_ptr,
    PictureParentControlSet         *pcs_tpl_base_ptr,
    EbBool                          *trailing_frames) {
    uint64_t curr_poc = pcs_tpl_group_frame_ptr->picture_number;
    uint64_t base_poc = pcs_tpl_base_ptr->picture_number;

    // 17/18/19, which is out side of the minigop
    *trailing_frames = (!pcs_tpl_base_ptr->idr_flag) && (curr_poc > base_poc);

    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref0_count = 0;
    pcs_tpl_group_frame_ptr->tpl_data.tpl_ref1_count = 0;
    EB_MEMSET(pcs_tpl_group_frame_ptr->tpl_data.ref_in_slide_window,
        0,
        MAX_NUM_OF_REF_PIC_LIST*REF_LIST_MAX_DEPTH * sizeof(EbBool));
    EB_MEMSET(pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[REF_LIST_0],
            0,
            REF_LIST_MAX_DEPTH * sizeof(EbDownScaledBufDescPtrArray));
    EB_MEMSET(pcs_tpl_group_frame_ptr->tpl_data.tpl_ref_ds_ptr_array[REF_LIST_1],
            0,
            REF_LIST_MAX_DEPTH * sizeof(EbDownScaledBufDescPtrArray));
    if (*trailing_frames) {
        // If trailing frames, set as B slice and layer1 first, will update later when getting the refs
        pcs_tpl_group_frame_ptr->tpl_data.tpl_slice_type = B_SLICE;

    } else {
        pcs_tpl_group_frame_ptr->tpl_data.tpl_slice_type = pcs_tpl_group_frame_ptr->slice_type;
        pcs_tpl_group_frame_ptr->tpl_data.tpl_temporal_layer_index = pcs_tpl_group_frame_ptr->temporal_layer_index;
        pcs_tpl_group_frame_ptr->tpl_data.is_used_as_reference_flag = pcs_tpl_group_frame_ptr->is_used_as_reference_flag;
        pcs_tpl_group_frame_ptr->tpl_data.tpl_decode_order = pcs_tpl_group_frame_ptr->decode_order;
    }
    return 0;
}


 EbErrorType tpl_get_open_loop_me(
    PictureManagerContext           *context_ptr,
    SequenceControlSet              *scs_ptr,
    PictureParentControlSet         *pcs_tpl_base_ptr) {

    if (scs_ptr->static_config.enable_tpl_la &&
        pcs_tpl_base_ptr->temporal_layer_index == 0) {
        // Do tpl ME to get ME results
        for (uint32_t i = 0; i < pcs_tpl_base_ptr->tpl_group_size; i++) {
            PictureParentControlSet* pcs_tpl_group_frame_ptr = pcs_tpl_base_ptr->tpl_group[i];
            uint8_t ref_list0_count = 0;
            uint8_t ref_list1_count = 0;
            EbBool  is_trailing_tpl_frame = EB_FALSE;
            tpl_init_pcs_tpl_data(pcs_tpl_group_frame_ptr,
                pcs_tpl_base_ptr,
                &is_trailing_tpl_frame);

            if (pcs_tpl_group_frame_ptr->tpl_data.tpl_slice_type != I_SLICE) {

                tpl_setup_me_refs(scs_ptr,
                        pcs_tpl_group_frame_ptr, pcs_tpl_base_ptr,
                        &ref_list0_count, &ref_list1_count,
                        &is_trailing_tpl_frame);

                if (scs_ptr->in_loop_me == 0)
                continue;
                if (!pcs_tpl_group_frame_ptr->tpl_me_done){
                    // Initialize Segments
                    pcs_tpl_group_frame_ptr->tpl_me_segments_column_count = 1;//scs_ptr->tf_segment_column_count;
                    pcs_tpl_group_frame_ptr->tpl_me_segments_row_count = 1;//scs_ptr->tf_segment_row_count;
                    pcs_tpl_group_frame_ptr->tpl_me_segments_total_count =
                        (uint16_t)(pcs_tpl_group_frame_ptr->tpl_me_segments_column_count *
                                   pcs_tpl_group_frame_ptr->tpl_me_segments_row_count);
                    pcs_tpl_group_frame_ptr->tpl_me_seg_acc = 0;

                    for (int16_t seg_idx = 0; seg_idx < pcs_tpl_group_frame_ptr->tpl_me_segments_total_count; ++seg_idx) {
                        EbObjectWrapper *out_results_wrapper_ptr;

                        svt_get_empty_object(
                                context_ptr->picture_manager_output_fifo_ptr,
                                &out_results_wrapper_ptr);

                        PictureManagerResults *out_results_ptr = (PictureManagerResults*)out_results_wrapper_ptr->object_ptr;
                        out_results_ptr->pcs_wrapper_ptr = pcs_tpl_group_frame_ptr->p_pcs_wrapper_ptr;
                        out_results_ptr->segment_index = seg_idx;
                        out_results_ptr->task_type = 2;
                        out_results_ptr->tpl_ref_list0_count = ref_list0_count;
                        out_results_ptr->tpl_ref_list1_count = ref_list1_count;
                        out_results_ptr->temporal_layer_index = is_trailing_tpl_frame ? 1 : pcs_tpl_group_frame_ptr->temporal_layer_index;
                        out_results_ptr->is_used_as_reference_flag = is_trailing_tpl_frame ? 0 : pcs_tpl_group_frame_ptr->is_used_as_reference_flag;
                        svt_post_full_object(out_results_wrapper_ptr);
                    }

                    svt_block_on_semaphore(pcs_tpl_group_frame_ptr->tpl_me_done_semaphore); //chkn we can do all in // ?

                    // When TPL16, flag tpl_me_done of 17/18/19 will be set done during TPL32
                    if (!is_trailing_tpl_frame) {
                        pcs_tpl_group_frame_ptr->tpl_me_done = 1;
                    }
                }
            }
        }
    }
    // Release pa reference
    if (scs_ptr->in_loop_me)
        release_pa_reference_objects(scs_ptr, pcs_tpl_base_ptr);
    return 0;
}

void init_enc_dec_segement(PictureParentControlSet *parentpicture_control_set_ptr) {
    SequenceControlSet *scs_ptr = (SequenceControlSet *)parentpicture_control_set_ptr->scs_wrapper_ptr->object_ptr;
    uint8_t pic_width_in_sb = (uint8_t)((parentpicture_control_set_ptr->aligned_width +
                scs_ptr->sb_size_pix - 1) /
            scs_ptr->sb_size_pix);
    uint8_t picture_height_in_sb = (uint8_t)((parentpicture_control_set_ptr->aligned_height +
                scs_ptr->sb_size_pix - 1) /
            scs_ptr->sb_size_pix);
    set_tile_info(parentpicture_control_set_ptr);
    int sb_size_log2 = scs_ptr->seq_header.sb_size_log2;
    uint32_t enc_dec_seg_col_cnt = scs_ptr->enc_dec_segment_col_count_array[parentpicture_control_set_ptr->temporal_layer_index];
    uint32_t enc_dec_seg_row_cnt = scs_ptr->enc_dec_segment_row_count_array[parentpicture_control_set_ptr->temporal_layer_index];
    Av1Common *const cm = parentpicture_control_set_ptr->av1_cm;
    const int tile_cols = parentpicture_control_set_ptr->av1_cm->tiles_info.tile_cols;
    const int tile_rows = parentpicture_control_set_ptr->av1_cm->tiles_info.tile_rows;
    uint8_t tile_group_cols = MIN(
            tile_cols,
            scs_ptr
            ->tile_group_col_count_array[parentpicture_control_set_ptr->temporal_layer_index]);
    uint8_t tile_group_rows = MIN(
            tile_rows,
            scs_ptr
            ->tile_group_row_count_array[parentpicture_control_set_ptr->temporal_layer_index]);

    if (tile_group_cols * tile_group_rows > 1) {
        enc_dec_seg_col_cnt = MIN(enc_dec_seg_col_cnt,
                (uint8_t)(pic_width_in_sb / tile_group_cols));
        enc_dec_seg_row_cnt = MIN(
                enc_dec_seg_row_cnt,
                (uint8_t)(picture_height_in_sb / tile_group_rows));
    }
    parentpicture_control_set_ptr->tile_group_cols = tile_group_cols;
    parentpicture_control_set_ptr->tile_group_rows = tile_group_rows;

    uint8_t tile_group_col_start_tile_idx[1024];
    uint8_t tile_group_row_start_tile_idx[1024];

    // Get the tile start index for tile group
    for (uint8_t c = 0; c <= tile_group_cols; c++) {
        tile_group_col_start_tile_idx[c] = c * tile_cols / tile_group_cols;
    }
    for (uint8_t r = 0; r <= tile_group_rows; r++) {
        tile_group_row_start_tile_idx[r] = r * tile_rows / tile_group_rows;
    }
    for (uint8_t r = 0; r < tile_group_rows; r++) {
        for (uint8_t c = 0; c < tile_group_cols; c++) {
            uint16_t tile_group_idx        = r * tile_group_cols + c;
            uint16_t top_left_tile_col_idx = tile_group_col_start_tile_idx[c];
            uint16_t top_left_tile_row_idx = tile_group_row_start_tile_idx[r];
            uint16_t bottom_right_tile_col_idx =
                tile_group_col_start_tile_idx[c + 1];
            uint16_t bottom_right_tile_row_idx =
                tile_group_row_start_tile_idx[r + 1];

            TileGroupInfo *tg_info_ptr =
                &parentpicture_control_set_ptr->tile_group_info[tile_group_idx];

            tg_info_ptr->tile_group_tile_start_x = top_left_tile_col_idx;
            tg_info_ptr->tile_group_tile_end_x   = bottom_right_tile_col_idx;

            tg_info_ptr->tile_group_tile_start_y = top_left_tile_row_idx;
            tg_info_ptr->tile_group_tile_end_y   = bottom_right_tile_row_idx;

            tg_info_ptr->tile_group_sb_start_x =
                cm->tiles_info.tile_col_start_mi[top_left_tile_col_idx] >>
                sb_size_log2;
            tg_info_ptr->tile_group_sb_start_y =
                cm->tiles_info.tile_row_start_mi[top_left_tile_row_idx] >>
                sb_size_log2;

            // Get the SB end of the bottom right tile
            tg_info_ptr->tile_group_sb_end_x =
                (cm->tiles_info.tile_col_start_mi[bottom_right_tile_col_idx] >>
                 sb_size_log2);
            tg_info_ptr->tile_group_sb_end_y =
                (cm->tiles_info.tile_row_start_mi[bottom_right_tile_row_idx] >>
                 sb_size_log2);

            // Get the width/height of tile group in SB
            tg_info_ptr->tile_group_height_in_sb =
                tg_info_ptr->tile_group_sb_end_y -
                tg_info_ptr->tile_group_sb_start_y;
            tg_info_ptr->tile_group_width_in_sb =
                tg_info_ptr->tile_group_sb_end_x -
                tg_info_ptr->tile_group_sb_start_x;

            // Init segments within the tile group
            enc_dec_segments_init(
                    parentpicture_control_set_ptr->child_pcs->enc_dec_segment_ctrl[tile_group_idx],
                    enc_dec_seg_col_cnt,
                    enc_dec_seg_row_cnt,
                    tg_info_ptr->tile_group_width_in_sb,
                    tg_info_ptr->tile_group_height_in_sb);
            // Enable tile parallelism in Entropy Coding stage
            for (uint16_t s = top_left_tile_row_idx;
                    s < bottom_right_tile_row_idx;
                    s++) {
                for (uint16_t d = top_left_tile_col_idx;
                        d < bottom_right_tile_col_idx;
                        d++) {
                    uint16_t tileIdx = s * tile_cols + d;
                    parentpicture_control_set_ptr->child_pcs->entropy_coding_info[tileIdx]
                        ->entropy_coding_current_row = 0;
                    parentpicture_control_set_ptr->child_pcs->entropy_coding_info[tileIdx]
                        ->entropy_coding_current_available_row = 0;
                    parentpicture_control_set_ptr->child_pcs->entropy_coding_info[tileIdx]
                        ->entropy_coding_row_count =
                        (cm->tiles_info.tile_row_start_mi[s + 1] -
                         cm->tiles_info.tile_row_start_mi[s]) >>
                        sb_size_log2;
                    parentpicture_control_set_ptr->child_pcs->entropy_coding_info[tileIdx]
                        ->entropy_coding_in_progress = EB_FALSE;
                    parentpicture_control_set_ptr->child_pcs->entropy_coding_info[tileIdx]
                        ->entropy_coding_tile_done = EB_FALSE;

                    for (unsigned rowIndex = 0; rowIndex < MAX_SB_ROWS;
                            ++rowIndex) {
                        parentpicture_control_set_ptr->child_pcs->entropy_coding_info[tileIdx]
                            ->entropy_coding_row_array[rowIndex] = EB_FALSE;
                    }
                }
            }
            parentpicture_control_set_ptr->child_pcs->entropy_coding_pic_reset_flag = EB_TRUE;
        }
    }
}
/* Picture Manager Kernel */

/***************************************************************************************************
 *
 * @brief
 *  The Picture Manager Process performs the function of managing both the Input Picture buffers and
 *  the Reference Picture buffers and subdividing the Input Picture into Tiles.
 *
 * @par Description:
 *  Both the Input Picture and Reference Picture buffers particular management depends on
 *  the GoP structure already implemented in the Picture decision. Also note that the
 *  Picture Manager sets up the RPS for Entropy Coding as well.
 *
 * @param[in] Input Picture
 *  Input Picture Data
 *
 * @param[in] Reference Picture
 *  Reference Picture Data
 *
 * @param[out] Picture Control Set
 *  Picture Control Set with fully available Reference List
 *
 ***************************************************************************************************/
void *picture_manager_kernel(void *input_ptr) {
    EbThreadContext *      thread_context_ptr = (EbThreadContext *)input_ptr;
    PictureManagerContext *context_ptr        = (PictureManagerContext *)thread_context_ptr->priv;

    EbObjectWrapper *        child_pcs_wrapper_ptr;
    PictureControlSet *      child_pcs_ptr;
    PictureParentControlSet *pcs_ptr;
    SequenceControlSet *     scs_ptr;
    EncodeContext *          encode_context_ptr;

    EbObjectWrapper *    input_picture_demux_wrapper_ptr;
    PictureDemuxResults *input_picture_demux_ptr;

    EbBool availability_flag;

    PredictionStructureEntry *pred_position_ptr;
    InputQueueEntry *         input_entry_ptr;
    uint32_t                  input_queue_index;
    uint64_t                  current_input_poc;
    ReferenceQueueEntry *     reference_entry_ptr;
    uint32_t                  reference_queue_index;
    uint64_t                  ref_poc;
    uint32_t                  dep_idx;
    PictureParentControlSet * entry_pcs_ptr;
    SequenceControlSet *      entry_scs_ptr;

    // Initialization
    uint8_t                     pic_width_in_sb;
    uint8_t                     picture_height_in_sb;
    uint64_t decode_order = 0;
    // Debug
    uint32_t loop_count = 0;

    for (;;) {
        // Get Input Full Object
        EB_GET_FULL_OBJECT(context_ptr->picture_input_fifo_ptr, &input_picture_demux_wrapper_ptr);

        input_picture_demux_ptr =
            (PictureDemuxResults *)input_picture_demux_wrapper_ptr->object_ptr;

        // *Note - This should be overhauled and/or replaced when we
        //   need hierarchical support.
        loop_count++;

        switch (input_picture_demux_ptr->picture_type) {
        case EB_PIC_INPUT:

            pcs_ptr =
                (PictureParentControlSet *)input_picture_demux_ptr->pcs_wrapper_ptr->object_ptr;
            scs_ptr            = (SequenceControlSet *)pcs_ptr->scs_wrapper_ptr->object_ptr;
            encode_context_ptr = scs_ptr->encode_context_ptr;

            //SVT_LOG("\nPicture Manager Process @ %d \n ", pcs_ptr->picture_number);
                pred_position_ptr = pcs_ptr->pred_struct_ptr
                                        ->pred_struct_entry_ptr_array[pcs_ptr->pred_struct_index];

                copy_dep_cnt_cleaning_list(encode_context_ptr, pcs_ptr);

                clean_pictures_in_ref_queue(encode_context_ptr);

                CHECK_REPORT_ERROR(
                    (((encode_context_ptr->input_picture_queue_head_index !=
                       encode_context_ptr->input_picture_queue_tail_index) ||
                      (encode_context_ptr
                           ->input_picture_queue[encode_context_ptr->input_picture_queue_head_index]
                           ->input_object_ptr == NULL))),
                    encode_context_ptr->app_callback_ptr,
                    EB_ENC_PM_ERROR4);

                // Place Picture in input queue
                input_entry_ptr =
                    encode_context_ptr
                        ->input_picture_queue[encode_context_ptr->input_picture_queue_tail_index];
                input_entry_ptr->input_object_ptr = input_picture_demux_ptr->pcs_wrapper_ptr;
                encode_context_ptr->input_picture_queue_tail_index =
                    (encode_context_ptr->input_picture_queue_tail_index ==
                     INPUT_QUEUE_MAX_DEPTH - 1)
                        ? 0
                        : encode_context_ptr->input_picture_queue_tail_index + 1;

                // Copy the reference lists into the inputEntry and
                // set the Reference Counts Based on Temporal Layer and how many frames are active
                input_entry_ptr->list0_ptr = &pred_position_ptr->ref_list0;
                input_entry_ptr->list1_ptr = &pred_position_ptr->ref_list1;
                if (!pcs_ptr->is_overlay) {
                    // Check if the ReferencePictureQueue is full.
                    CHECK_REPORT_ERROR(
                        (((encode_context_ptr->reference_picture_queue_head_index !=
                           encode_context_ptr->reference_picture_queue_tail_index) ||
                          (encode_context_ptr
                               ->reference_picture_queue[encode_context_ptr
                                                             ->reference_picture_queue_head_index]
                               ->reference_object_ptr == NULL))),
                        encode_context_ptr->app_callback_ptr,
                        EB_ENC_PM_ERROR5);

                    // Create Reference Queue Entry even if picture will not be referenced
                    reference_entry_ptr =
                        encode_context_ptr->reference_picture_queue
                            [encode_context_ptr->reference_picture_queue_tail_index];
                    reference_entry_ptr->picture_number        = pcs_ptr->picture_number;
                    reference_entry_ptr->reference_object_ptr  = (EbObjectWrapper *)NULL;
                    reference_entry_ptr->ref_wraper = pcs_ptr->reference_picture_wrapper_ptr; //at this time only the src data is valid
                    reference_entry_ptr->release_enable        = EB_TRUE;
                    reference_entry_ptr->reference_available   = EB_FALSE;
                    reference_entry_ptr->slice_type            = pcs_ptr->slice_type;
                    reference_entry_ptr->temporal_layer_index  = pcs_ptr->temporal_layer_index;
                    reference_entry_ptr->frame_context_updated = EB_FALSE;
                    reference_entry_ptr->is_alt_ref            = pcs_ptr->is_alt_ref;
                    reference_entry_ptr->feedback_arrived      = EB_FALSE;
                    reference_entry_ptr->is_used_as_reference_flag =
                        pcs_ptr->is_used_as_reference_flag;
                    encode_context_ptr->reference_picture_queue_tail_index =
                        (encode_context_ptr->reference_picture_queue_tail_index ==
                         REFERENCE_QUEUE_MAX_DEPTH - 1)
                            ? 0
                            : encode_context_ptr->reference_picture_queue_tail_index + 1;

                    // Copy the Dependent Lists
                    // *Note - we are removing any leading picture dependencies for now
                    reference_entry_ptr->list0.list_count = 0;
                    for (dep_idx = 0; dep_idx < pred_position_ptr->dep_list0.list_count;
                         ++dep_idx) {
                        if (pred_position_ptr->dep_list0.list[dep_idx] >= 0)
                            reference_entry_ptr->list0
                                .list[reference_entry_ptr->list0.list_count++] =
                                pred_position_ptr->dep_list0.list[dep_idx];
                    }

                    reference_entry_ptr->list1.list_count = pred_position_ptr->dep_list1.list_count;
                    for (dep_idx = 0; dep_idx < pred_position_ptr->dep_list1.list_count; ++dep_idx)
                        reference_entry_ptr->list1.list[dep_idx] =
                            pred_position_ptr->dep_list1.list[dep_idx];
                    reference_entry_ptr->dep_list0_count =
                        (pcs_ptr->is_alt_ref) ? reference_entry_ptr->list0.list_count + 1
                                              : reference_entry_ptr->list0.list_count;

                    reference_entry_ptr->dep_list1_count = reference_entry_ptr->list1.list_count;
                    reference_entry_ptr->dependent_count =
                        reference_entry_ptr->dep_list0_count + reference_entry_ptr->dep_list1_count;
                    //remove dependency to alredy knowmn broken links from PD
                    reference_entry_ptr->dependent_count += pcs_ptr->self_updated_links;

                    CHECK_REPORT_ERROR(
                        (pcs_ptr->pred_struct_ptr->pred_struct_period * REF_LIST_MAX_DEPTH <
                         MAX_ELAPSED_IDR_COUNT),
                        encode_context_ptr->app_callback_ptr,
                        EB_ENC_PM_ERROR6);
                }
            break;

        case EB_PIC_REFERENCE:

            scs_ptr = (SequenceControlSet *)input_picture_demux_ptr->scs_wrapper_ptr->object_ptr;
            encode_context_ptr = scs_ptr->encode_context_ptr;
            clean_pictures_in_ref_queue(scs_ptr->encode_context_ptr);
            ((EbReferenceObject *)input_picture_demux_ptr->reference_picture_wrapper_ptr->object_ptr)->ds_pics.picture_number =
                input_picture_demux_ptr->picture_number;
            // Check if Reference Queue is full
            CHECK_REPORT_ERROR((encode_context_ptr->reference_picture_queue_head_index !=
                                encode_context_ptr->reference_picture_queue_tail_index),
                               encode_context_ptr->app_callback_ptr,
                               EB_ENC_PM_ERROR7);

            reference_queue_index = encode_context_ptr->reference_picture_queue_head_index;
            // Find the Reference in the Reference Queue
            do {
                reference_entry_ptr =
                    encode_context_ptr->reference_picture_queue[reference_queue_index];

                if (reference_entry_ptr->picture_number ==
                    input_picture_demux_ptr->picture_number) {
                    // Assign the reference object if there is a match
                    reference_entry_ptr->reference_object_ptr =
                        input_picture_demux_ptr->reference_picture_wrapper_ptr;

                    // Set the reference availability
                    reference_entry_ptr->reference_available = EB_TRUE;
                }

                // Increment the reference_queue_index Iterator
                reference_queue_index = (reference_queue_index == REFERENCE_QUEUE_MAX_DEPTH - 1)
                                            ? 0
                                            : reference_queue_index + 1;
            } while (
                (reference_queue_index != encode_context_ptr->reference_picture_queue_tail_index) &&
                (reference_entry_ptr->picture_number != input_picture_demux_ptr->picture_number));

            CHECK_REPORT_ERROR(
                (reference_entry_ptr->picture_number == input_picture_demux_ptr->picture_number),
                encode_context_ptr->app_callback_ptr,
                EB_ENC_PM_ERROR8);

            //keep the relase of SCS here because we still need the encodeContext strucutre here
            // Release the Reference's SequenceControlSet
            svt_release_object(input_picture_demux_ptr->scs_wrapper_ptr);

            break;
        case EB_PIC_FEEDBACK:
            scs_ptr = (SequenceControlSet *)input_picture_demux_ptr->scs_wrapper_ptr->object_ptr;
            encode_context_ptr    = scs_ptr->encode_context_ptr;

            clean_pictures_in_ref_queue(scs_ptr->encode_context_ptr);
            reference_queue_index = encode_context_ptr->reference_picture_queue_head_index;
            // Find the Reference in the Reference Queue
            do {
                reference_entry_ptr =
                    encode_context_ptr->reference_picture_queue[reference_queue_index];
                if (reference_entry_ptr->picture_number ==
                    input_picture_demux_ptr->picture_number) {
                    // Set the feedback arrived
                    reference_entry_ptr->feedback_arrived      = EB_TRUE;
                    reference_entry_ptr->frame_context_updated = EB_TRUE;
                }
                // Increment the reference_queue_index Iterator
                reference_queue_index = (reference_queue_index == REFERENCE_QUEUE_MAX_DEPTH - 1)
                                            ? 0
                                            : reference_queue_index + 1;

            } while (
                (reference_queue_index != encode_context_ptr->reference_picture_queue_tail_index) &&
                (reference_entry_ptr->picture_number != input_picture_demux_ptr->picture_number));
            // Update the last decode order
            if(input_picture_demux_ptr->decode_order == decode_order)
                decode_order++;

            //keep the release of SCS here because we still need the encodeContext structure here
            // Release the Reference's SequenceControlSet
            svt_release_object(input_picture_demux_ptr->scs_wrapper_ptr);

            break;
        default:

            scs_ptr = (SequenceControlSet *)input_picture_demux_ptr->scs_wrapper_ptr->object_ptr;
            encode_context_ptr = scs_ptr->encode_context_ptr;

            CHECK_REPORT_ERROR_NC(encode_context_ptr->app_callback_ptr, EB_ENC_PM_ERROR9);

            pcs_ptr            = (PictureParentControlSet *)NULL;
            encode_context_ptr = (EncodeContext *)NULL;

            break;
        }

        // ***********************************
        //  Common Code
        // *************************************

        // Walk the input queue and start all ready pictures.  Mark entry as null after started.  Increment the head as you go.
        if (encode_context_ptr != (EncodeContext *)NULL) {
            input_queue_index = encode_context_ptr->input_picture_queue_head_index;
            while (input_queue_index != encode_context_ptr->input_picture_queue_tail_index) {
                input_entry_ptr = encode_context_ptr->input_picture_queue[input_queue_index];

                if (input_entry_ptr->input_object_ptr != NULL) {
                    entry_pcs_ptr =
                        (PictureParentControlSet *)input_entry_ptr->input_object_ptr->object_ptr;
                    entry_scs_ptr =
                        (SequenceControlSet *)entry_pcs_ptr->scs_wrapper_ptr->object_ptr;

                    availability_flag = EB_TRUE;
                    if (entry_pcs_ptr->decode_order != decode_order &&
                    (scs_ptr->enable_dec_order || use_input_stat(scs_ptr) || scs_ptr->lap_enabled ))
                        availability_flag = EB_FALSE;

                    //pic mgr starts pictures in dec order (no need to wait for feedback)
                    if(entry_scs_ptr->enable_pic_mgr_dec_order)
                        if (entry_pcs_ptr->picture_number > 0 && entry_pcs_ptr->decode_order != context_ptr->pmgr_dec_order + 1)
                            availability_flag = EB_FALSE;
                    // Check RefList0 Availability
                    for (uint8_t ref_idx = 0; ref_idx < entry_pcs_ptr->ref_list0_count; ++ref_idx) {
                        //if (entry_pcs_ptr->ref_list0_count)  // NM: to double check.
                        {

                            if (entry_pcs_ptr->is_overlay)
                                // hardcode the reference for the overlay frame
                                ref_poc = entry_pcs_ptr->picture_number;
                            else
                                ref_poc = POC_CIRCULAR_ADD(
                                    entry_pcs_ptr->picture_number,
                                    -input_entry_ptr->list0_ptr->reference_list[ref_idx]);

                            reference_entry_ptr = search_ref_in_ref_queue(encode_context_ptr, ref_poc);

                            // Increment the current_input_poc is the case of POC rollover
                            current_input_poc = encode_context_ptr->current_input_poc;
                            //current_input_poc += ((current_input_poc < ref_poc) && (input_entry_ptr->list0_ptr->reference_list[ref_idx] > 0)) ?
                            //    (1 << entry_scs_ptr->bits_for_picture_order_count) :
                            //    0;
                            if (reference_entry_ptr != NULL) {
                            availability_flag =
                                (availability_flag == EB_FALSE) ? EB_FALSE
                                                                : // Don't update if already False
                                    (ref_poc > current_input_poc)
                                        ? EB_FALSE
                                        : // The Reference has not been received as an Input Picture yet, then its availability is false
/*AAA READ2*/                                        ( 
                                         (scs_ptr->static_config.rate_control_mode &&
                                          entry_pcs_ptr->slice_type != I_SLICE &&
                                          entry_pcs_ptr->temporal_layer_index == 0 &&
                                          !reference_entry_ptr->feedback_arrived) &&
    !encode_context_ptr->terminating_sequence_flag_received)
                                            ? EB_FALSE
                                            : (entry_pcs_ptr->frame_end_cdf_update_mode &&
                                               !reference_entry_ptr->frame_context_updated)
                                                  ? EB_FALSE
                                                  : (reference_entry_ptr->reference_available)
                                                        ? EB_TRUE
                                                        : // The Reference has been completed
                                                        EB_FALSE; // The Reference has not been completed
                            }else{
                                availability_flag = EB_FALSE;
                            }
                        }
                    }
                    // Check RefList1 Availability
                    if (entry_pcs_ptr->slice_type == B_SLICE) {
                        for (uint8_t ref_idx = 0; ref_idx < entry_pcs_ptr->ref_list1_count;
                             ++ref_idx) {
                            // if (entry_pcs_ptr->ref_list1_count) // NM: To double check
                            {
                                // If Reference is valid (non-zero), update the availability
                                if (input_entry_ptr->list1_ptr->reference_list[ref_idx] !=
                                    (int32_t)INVALID_POC) {

                                    ref_poc = POC_CIRCULAR_ADD(
                                        entry_pcs_ptr->picture_number,
                                        -input_entry_ptr->list1_ptr->reference_list[ref_idx]);

                                    reference_entry_ptr = search_ref_in_ref_queue(encode_context_ptr, ref_poc);


                                    // Increment the current_input_poc is the case of POC rollover
                                    current_input_poc = encode_context_ptr->current_input_poc;
                                    //current_input_poc += ((current_input_poc < ref_poc && input_entry_ptr->list1_ptr->reference_list[ref_idx] > 0)) ?
                                    //    (1 << entry_scs_ptr->bits_for_picture_order_count) :
                                    //    0;

                                    if (reference_entry_ptr != NULL){
                                    availability_flag =
                                        (availability_flag == EB_FALSE)
                                            ? EB_FALSE
                                            : // Don't update if already False
                                            (ref_poc > current_input_poc)
                                                ? EB_FALSE
                                                : // The Reference has not been received as an Input Picture yet, then its availability is false
                                                (
                                                 (scs_ptr->static_config.rate_control_mode &&
                                                  entry_pcs_ptr->slice_type != I_SLICE &&
                                                  entry_pcs_ptr->temporal_layer_index == 0 &&
                                                  !reference_entry_ptr->feedback_arrived && !encode_context_ptr
                                                      ->terminating_sequence_flag_received))
                                                    ? EB_FALSE
                                                    : (entry_pcs_ptr->frame_end_cdf_update_mode &&
                                                       !reference_entry_ptr->frame_context_updated)
                                                          ? EB_FALSE
                                                          : (reference_entry_ptr
                                                                 ->reference_available)
                                                                ? EB_TRUE
                                                                : // The Reference has been completed
                                                                EB_FALSE; // The Reference has not been completed

                                    }else{
                                        availability_flag = EB_FALSE; }
                                }
                            }
                        }
                    }

                    if (availability_flag == EB_TRUE) {
                        // Get New  Empty Child PCS from PCS Pool
                        svt_get_empty_object(context_ptr->picture_control_set_fifo_ptr,
                                            &child_pcs_wrapper_ptr);

                        // Child PCS is released by Packetization
                        svt_object_inc_live_count(child_pcs_wrapper_ptr, 1);

                        child_pcs_ptr = (PictureControlSet *)child_pcs_wrapper_ptr->object_ptr;

                        child_pcs_ptr->c_pcs_wrapper_ptr = child_pcs_wrapper_ptr;

                        //1.Link The Child PCS to its Parent
                        child_pcs_ptr->picture_parent_control_set_wrapper_ptr =
                            input_entry_ptr->input_object_ptr;
                        child_pcs_ptr->parent_pcs_ptr = entry_pcs_ptr;

                        child_pcs_ptr->parent_pcs_ptr->child_pcs = child_pcs_ptr;

                        //2. Have some common information between  ChildPCS and ParentPCS.
                        child_pcs_ptr->scs_wrapper_ptr      = entry_pcs_ptr->scs_wrapper_ptr;
                        child_pcs_ptr->picture_qp           = entry_pcs_ptr->picture_qp;
                        child_pcs_ptr->picture_number       = entry_pcs_ptr->picture_number;
                        child_pcs_ptr->slice_type           = entry_pcs_ptr->slice_type;
                        child_pcs_ptr->temporal_layer_index = entry_pcs_ptr->temporal_layer_index;

                        child_pcs_ptr->parent_pcs_ptr->total_num_bits = 0;
                        child_pcs_ptr->parent_pcs_ptr->picture_qp     = entry_pcs_ptr->picture_qp;
                        child_pcs_ptr->parent_pcs_ptr->sad_me         = 0;
                        child_pcs_ptr->parent_pcs_ptr->quantized_coeff_num_bits = 0;
                        child_pcs_ptr->enc_mode = entry_pcs_ptr->enc_mode;
                        child_pcs_ptr->sb_total_count = entry_pcs_ptr->sb_total_count;

                        child_pcs_ptr->enc_dec_coded_sb_count = 0;
                        child_pcs_ptr->parent_pcs_ptr->av1_cm->rst_tmpbuf = child_pcs_ptr->rst_tmpbuf;

                        context_ptr->pmgr_dec_order = child_pcs_ptr->parent_pcs_ptr->decode_order;
                        //3.make all  init for ChildPCS
                        pic_width_in_sb = (uint8_t)((entry_pcs_ptr->aligned_width +
                                                     entry_scs_ptr->sb_size_pix - 1) /
                                                    entry_scs_ptr->sb_size_pix);
                        picture_height_in_sb =
                            (uint8_t)((entry_pcs_ptr->aligned_height +
                                       entry_scs_ptr->sb_size_pix - 1) /
                                      entry_scs_ptr->sb_size_pix);

                        init_enc_dec_segement(entry_pcs_ptr);

                        int      sb_size_log2    = entry_scs_ptr->seq_header.sb_size_log2;
                        struct PictureParentControlSet *ppcs_ptr = child_pcs_ptr->parent_pcs_ptr;
                        const int tile_cols = ppcs_ptr->av1_cm->tiles_info.tile_cols;
                        const int tile_rows = ppcs_ptr->av1_cm->tiles_info.tile_rows;
                        Av1Common *const                cm       = ppcs_ptr->av1_cm;
                        uint16_t                        tile_row, tile_col;
                        uint32_t                        x_sb_index, y_sb_index;
                        TileInfo  tile_info;

                        child_pcs_ptr->sb_total_count_pix = pic_width_in_sb * picture_height_in_sb;

                        if(entry_pcs_ptr->frame_superres_enabled){
                            // Modify sb_prt_array in child pcs
                            uint16_t    sb_index;
                            uint16_t    sb_origin_x = 0;
                            uint16_t    sb_origin_y = 0;
                            for (sb_index = 0; sb_index < child_pcs_ptr->sb_total_count_pix; ++sb_index) {
                                largest_coding_unit_dctor(child_pcs_ptr->sb_ptr_array[sb_index]);
                                largest_coding_unit_ctor(child_pcs_ptr->sb_ptr_array[sb_index],
                                                         (uint8_t)scs_ptr->sb_size_pix,
                                                         (uint16_t)(sb_origin_x * scs_ptr->sb_size_pix),
                                                         (uint16_t)(sb_origin_y * scs_ptr->sb_size_pix),
                                                         (uint16_t)sb_index,
                                                         child_pcs_ptr);
                                // Increment the Order in coding order (Raster Scan Order)
                                sb_origin_y = (sb_origin_x == pic_width_in_sb - 1) ? sb_origin_y + 1 : sb_origin_y;
                                sb_origin_x = (sb_origin_x == pic_width_in_sb - 1) ? 0 : sb_origin_x + 1;
                            }
                        }

                        // Update pcs_ptr->mi_stride
                        child_pcs_ptr->mi_stride = pic_width_in_sb * (scs_ptr->sb_size_pix >> MI_SIZE_LOG2);

                        // copy buffer info from the downsampled picture to the input frame 16 bit buffer
                        if(entry_pcs_ptr->frame_superres_enabled && scs_ptr->static_config.encoder_bit_depth > EB_8BIT){
                            copy_buffer_info(entry_pcs_ptr->enhanced_downscaled_picture_ptr, child_pcs_ptr->input_frame16bit);
                        }

//                        child_pcs_ptr->parent_pcs_ptr->av1_cm->pcs_ptr = child_pcs_ptr;
                        // Palette
                        alloc_palette_tokens(scs_ptr, child_pcs_ptr);
                        TOKENEXTRA * pre_tok  = child_pcs_ptr->tile_tok[0][0];
                        unsigned int tile_tok = 0;
                        //Tile Loop
                        for (tile_row = 0; tile_row < tile_rows; tile_row++) {
                            svt_av1_tile_set_row(&tile_info, &cm->tiles_info, cm->mi_rows, tile_row);

                            for (tile_col = 0; tile_col < tile_cols; tile_col++) {
                                svt_av1_tile_set_col(
                                    &tile_info, &cm->tiles_info, cm->mi_cols, tile_col);
                                tile_info.tile_rs_index = tile_col + tile_row * tile_cols;

                                // Palette
                                if (child_pcs_ptr->tile_tok[0][0]) {
                                    child_pcs_ptr->tile_tok[tile_row][tile_col] =
                                        pre_tok + tile_tok;
                                    pre_tok = child_pcs_ptr->tile_tok[tile_row][tile_col];
                                    int tile_mb_rows =
                                        (tile_info.mi_row_end - tile_info.mi_row_start + 2) >> 2;
                                    int tile_mb_cols =
                                        (tile_info.mi_col_end - tile_info.mi_col_start + 2) >> 2;
                                    tile_tok = get_token_alloc(
                                        tile_mb_rows, tile_mb_cols, (sb_size_log2 + 2), 2);
                                }
                                for ((y_sb_index = cm->tiles_info.tile_row_start_mi[tile_row] >>
                                                   sb_size_log2);
                                     (y_sb_index <
                                      ((uint32_t)cm->tiles_info.tile_row_start_mi[tile_row + 1] >>
                                       sb_size_log2));
                                     y_sb_index++) {
                                    for ((x_sb_index = cm->tiles_info.tile_col_start_mi[tile_col] >>
                                                       sb_size_log2);
                                         (x_sb_index < ((uint32_t)cm->tiles_info
                                                            .tile_col_start_mi[tile_col + 1] >>
                                                        sb_size_log2));
                                         x_sb_index++) {
                                        int sb_index =
                                            (uint16_t)(x_sb_index + y_sb_index * pic_width_in_sb);
                                        child_pcs_ptr->sb_ptr_array[sb_index]->tile_info =
                                            tile_info;
                                    }
                                }
                            }
                        }
                        cm->mi_stride = child_pcs_ptr->mi_stride;
                        // Picture edges
                        configure_picture_edges(entry_scs_ptr, child_pcs_ptr);
                        // Error resilience related
                        child_pcs_ptr->colocated_pu_ref_list = REF_LIST_0; // to be modified

                        // Rate Control
                        child_pcs_ptr->dif_blk_delta_qp_depth =
                            (uint8_t)entry_scs_ptr->input_resolution <= INPUT_SIZE_1080p_RANGE ? 2 : 3;

                        // Reset the Reference Lists
                        EB_MEMSET(child_pcs_ptr->ref_pic_ptr_array[REF_LIST_0],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(EbObjectWrapper *));
                        EB_MEMSET(child_pcs_ptr->ref_pic_ptr_array[REF_LIST_1],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(EbObjectWrapper *));
                        EB_MEMSET(child_pcs_ptr->ref_pic_qp_array[REF_LIST_0],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(uint8_t));
                        EB_MEMSET(child_pcs_ptr->ref_pic_qp_array[REF_LIST_1],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(uint8_t));
                        EB_MEMSET(child_pcs_ptr->ref_slice_type_array[REF_LIST_0],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(EB_SLICE));
                        EB_MEMSET(child_pcs_ptr->ref_slice_type_array[REF_LIST_1],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(EB_SLICE));
                        EB_MEMSET(child_pcs_ptr->ref_pic_referenced_area_avg_array[REF_LIST_0],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(uint64_t));
                        EB_MEMSET(child_pcs_ptr->ref_pic_referenced_area_avg_array[REF_LIST_1],
                                  0,
                                  REF_LIST_MAX_DEPTH * sizeof(uint64_t));
                        EB_MEMSET(child_pcs_ptr->ref_pic_r0[REF_LIST_0],
                            0,
                            REF_LIST_MAX_DEPTH * sizeof(double));
                        EB_MEMSET(child_pcs_ptr->ref_pic_r0[REF_LIST_1],
                            0,
                            REF_LIST_MAX_DEPTH * sizeof(double));
                        int8_t max_temporal_index = -1, ref_index = 0;
                        // Configure List0
                        if ((entry_pcs_ptr->slice_type == P_SLICE) ||
                            (entry_pcs_ptr->slice_type == B_SLICE)) {
                            for (uint8_t ref_idx = 0; ref_idx < entry_pcs_ptr->ref_list0_count;
                                 ++ref_idx) {
                                if (entry_pcs_ptr->ref_list0_count) {

                                    if (entry_pcs_ptr->is_overlay)
                                        ref_poc = entry_pcs_ptr->picture_number;
                                    else
                                        ref_poc = POC_CIRCULAR_ADD(
                                            entry_pcs_ptr->picture_number,
                                            -input_entry_ptr->list0_ptr->reference_list[ref_idx]);

                                    reference_entry_ptr = search_ref_in_ref_queue(encode_context_ptr, ref_poc);
                                    assert(reference_entry_ptr != 0);
                                    CHECK_REPORT_ERROR((reference_entry_ptr),
                                            encode_context_ptr->app_callback_ptr,
                                            EB_ENC_PM_ERROR10);
                                    if (entry_pcs_ptr->frame_end_cdf_update_mode) {
                                        child_pcs_ptr->ref_frame_context[svt_get_ref_frame_type(
                                                                             REF_LIST_0, ref_idx) -
                                                                         LAST_FRAME] =
                                            ((EbReferenceObject *)reference_entry_ptr
                                                 ->reference_object_ptr->object_ptr)
                                                ->frame_context;
                                        if (max_temporal_index <
                                                (int8_t)reference_entry_ptr->temporal_layer_index &&
                                            (int8_t)reference_entry_ptr->temporal_layer_index <=
                                                child_pcs_ptr->temporal_layer_index) {
                                            max_temporal_index =
                                                (int8_t)reference_entry_ptr->temporal_layer_index;
                                            ref_index =
                                                svt_get_ref_frame_type(REF_LIST_0, ref_idx) -
                                                LAST_FRAME;
                                            for (int frame = LAST_FRAME; frame <= ALTREF_FRAME;
                                                 ++frame)
                                            {
                                                EbReferenceObject *ref =
                                                    (EbReferenceObject *)reference_entry_ptr
                                                     ->reference_object_ptr->object_ptr;

                                                child_pcs_ptr->ref_global_motion[frame] =
                                                        ref->slice_type != I_SLICE ?
                                                        ref->global_motion[frame] :
                                                        default_warp_params;
                                            }
                                        }
                                    }
                                    // Set the Reference Object
                                    child_pcs_ptr->ref_pic_ptr_array[REF_LIST_0][ref_idx] =
                                        reference_entry_ptr->reference_object_ptr;
                                    child_pcs_ptr->ref_pic_qp_array[REF_LIST_0][ref_idx] =
                                        (uint8_t)((EbReferenceObject *)reference_entry_ptr
                                                      ->reference_object_ptr->object_ptr)
                                            ->qp;
                                    child_pcs_ptr->ref_slice_type_array[REF_LIST_0][ref_idx] =
                                        ((EbReferenceObject *)
                                             reference_entry_ptr->reference_object_ptr->object_ptr)
                                            ->slice_type;
                                    child_pcs_ptr
                                        ->ref_pic_referenced_area_avg_array[REF_LIST_0][ref_idx] =
                                        ((EbReferenceObject *)
                                             reference_entry_ptr->reference_object_ptr->object_ptr)
                                            ->referenced_area_avg;
                                    child_pcs_ptr
                                        ->ref_pic_r0[REF_LIST_0][ref_idx] =
                                        ((EbReferenceObject *)
                                            reference_entry_ptr->reference_object_ptr->object_ptr)
                                        ->r0;
                                    // Increment the Reference's liveCount by the number of tiles in the input picture
                                    svt_object_inc_live_count(
                                        reference_entry_ptr->reference_object_ptr, 1);

                                    // Decrement the Reference's dependent_count Count
                                    --reference_entry_ptr->dependent_count;

                                    CHECK_REPORT_ERROR(
                                        (reference_entry_ptr->dependent_count != ~0u),
                                        encode_context_ptr->app_callback_ptr,
                                        EB_ENC_PM_ERROR1);
                                }
                            }
                            //fill the non used spots to be used in MFMV.
                            for (uint8_t ref_idx = entry_pcs_ptr->ref_list0_count; ref_idx < 4;
                                 ++ref_idx)
                                child_pcs_ptr->ref_pic_ptr_array[REF_LIST_0][ref_idx] =
                                    child_pcs_ptr->ref_pic_ptr_array[REF_LIST_0][0];

                            if (entry_pcs_ptr->ref_list1_count == 0) {
                                for (uint8_t ref_idx = entry_pcs_ptr->ref_list1_count; ref_idx < 3;
                                     ++ref_idx)
                                    child_pcs_ptr->ref_pic_ptr_array[REF_LIST_1][ref_idx] =
                                        child_pcs_ptr->ref_pic_ptr_array[REF_LIST_0][0];
                            }
                        }

                        // Configure List1
                        if (entry_pcs_ptr->slice_type == B_SLICE) {
                            for (uint8_t ref_idx = 0; ref_idx < entry_pcs_ptr->ref_list1_count;
                                 ++ref_idx) {
                                if (entry_pcs_ptr->ref_list1_count) {

                                    ref_poc = POC_CIRCULAR_ADD(
                                        entry_pcs_ptr->picture_number,
                                        -input_entry_ptr->list1_ptr->reference_list[ref_idx]);

                                    reference_entry_ptr = search_ref_in_ref_queue(encode_context_ptr, ref_poc);

                                    assert(reference_entry_ptr != 0);
                                    CHECK_REPORT_ERROR((reference_entry_ptr),
                                        encode_context_ptr->app_callback_ptr,
                                        EB_ENC_PM_ERROR10);
                                    if (entry_pcs_ptr->frame_end_cdf_update_mode) {
                                        child_pcs_ptr->ref_frame_context[svt_get_ref_frame_type(
                                                                             REF_LIST_1, ref_idx) -
                                                                         LAST_FRAME] =
                                            ((EbReferenceObject *)reference_entry_ptr
                                                 ->reference_object_ptr->object_ptr)
                                                ->frame_context;
                                        if (max_temporal_index <
                                                (int8_t)reference_entry_ptr->temporal_layer_index &&
                                            reference_entry_ptr->slice_type != I_SLICE &&
                                            (int8_t)reference_entry_ptr->temporal_layer_index <=
                                                child_pcs_ptr->temporal_layer_index) {

                                            max_temporal_index =
                                                (int8_t)reference_entry_ptr->temporal_layer_index;
                                            ref_index =
                                                svt_get_ref_frame_type(REF_LIST_1, ref_idx) -
                                                LAST_FRAME;
                                            for (int frame = LAST_FRAME; frame <= ALTREF_FRAME;
                                                 ++frame)
                                            {
                                                EbReferenceObject *ref =
                                                    (EbReferenceObject *)reference_entry_ptr
                                                     ->reference_object_ptr->object_ptr;

                                                child_pcs_ptr->ref_global_motion[frame] =
                                                        ref->slice_type != I_SLICE ?
                                                        ref->global_motion[frame] :
                                                        default_warp_params;
                                            }
                                        }
                                    }
                                    // Set the Reference Object
                                    child_pcs_ptr->ref_pic_ptr_array[REF_LIST_1][ref_idx] =
                                        reference_entry_ptr->reference_object_ptr;
                                    child_pcs_ptr->ref_pic_qp_array[REF_LIST_1][ref_idx] =
                                        (uint8_t)((EbReferenceObject *)reference_entry_ptr
                                                      ->reference_object_ptr->object_ptr)
                                            ->qp;
                                    child_pcs_ptr->ref_slice_type_array[REF_LIST_1][ref_idx] =
                                        ((EbReferenceObject *)
                                             reference_entry_ptr->reference_object_ptr->object_ptr)
                                            ->slice_type;
                                    child_pcs_ptr
                                        ->ref_pic_referenced_area_avg_array[REF_LIST_1][ref_idx] =
                                        ((EbReferenceObject *)
                                             reference_entry_ptr->reference_object_ptr->object_ptr)
                                            ->referenced_area_avg;
                                    child_pcs_ptr
                                        ->ref_pic_r0[REF_LIST_1][ref_idx] =
                                        ((EbReferenceObject *)
                                            reference_entry_ptr->reference_object_ptr->object_ptr)
                                        ->r0;
                                    // Increment the Reference's liveCount by the number of tiles in the input picture
                                    svt_object_inc_live_count(
                                        reference_entry_ptr->reference_object_ptr, 1);

                                    // Decrement the Reference's dependent_count Count
                                    --reference_entry_ptr->dependent_count;

                                    CHECK_REPORT_ERROR(
                                        (reference_entry_ptr->dependent_count != ~0u),
                                        encode_context_ptr->app_callback_ptr,
                                        EB_ENC_PM_ERROR1);
                                }
                            }
                            //fill the non used spots to be used in MFMV.
                            if (entry_pcs_ptr->ref_list1_count) {
                                for (uint8_t ref_idx = entry_pcs_ptr->ref_list1_count; ref_idx < 3;
                                     ++ref_idx)
                                    child_pcs_ptr->ref_pic_ptr_array[REF_LIST_1][ref_idx] =
                                        child_pcs_ptr->ref_pic_ptr_array[REF_LIST_1][0];
                            }
                        }

                        // Adjust the Slice-type if the Lists are Empty, but don't reset the Prediction Structure
                        entry_pcs_ptr->slice_type =
                            (entry_pcs_ptr->ref_list1_count > 0)
                                ? B_SLICE
                                : (entry_pcs_ptr->ref_list0_count > 0) ? P_SLICE : I_SLICE;
                        if (entry_pcs_ptr->frame_end_cdf_update_mode) {
                            if (entry_pcs_ptr->slice_type != I_SLICE)
                                child_pcs_ptr->parent_pcs_ptr->frm_hdr.primary_ref_frame =
                                    ref_index;
                            else
                                child_pcs_ptr->parent_pcs_ptr->frm_hdr.primary_ref_frame =
                                    PRIMARY_REF_NONE;
                            child_pcs_ptr->parent_pcs_ptr->refresh_frame_context =
                                REFRESH_FRAME_CONTEXT_BACKWARD;

                        } else {
                            child_pcs_ptr->parent_pcs_ptr->frm_hdr.primary_ref_frame =
                                PRIMARY_REF_NONE;
                            child_pcs_ptr->parent_pcs_ptr->refresh_frame_context =
                                REFRESH_FRAME_CONTEXT_DISABLED;
                        }
                        // Increment the sequenceControlSet Wrapper's live count by 1 for only the pictures which are used as reference
                        if (child_pcs_ptr->parent_pcs_ptr->is_used_as_reference_flag) {
                            svt_object_inc_live_count(child_pcs_ptr->parent_pcs_ptr->scs_wrapper_ptr,
                                                     1);
                        }

                        // Get TPL ME
                        if (scs_ptr->in_loop_me|| scs_ptr->static_config.enable_tpl_la == 0)
                        tpl_get_open_loop_me(context_ptr, scs_ptr, child_pcs_ptr->parent_pcs_ptr);

                        const uint32_t segment_counts =  child_pcs_ptr->parent_pcs_ptr->inloop_me_segments_total_count;
                        for (uint32_t segment_index = 0; segment_index < segment_counts; ++segment_index) {
                            EbObjectWrapper               *out_results_wrapper_ptr;
                            // Get Empty Results Object
                            svt_get_empty_object(
                                context_ptr->picture_manager_output_fifo_ptr,
                                &out_results_wrapper_ptr);

                            PictureManagerResults   *out_results_ptr = (PictureManagerResults*)out_results_wrapper_ptr->object_ptr;
                            // PM output ppcs to iME kernel
                            out_results_ptr->pcs_wrapper_ptr = child_pcs_ptr->parent_pcs_ptr->p_pcs_wrapper_ptr;
                            out_results_ptr->segment_index = segment_index;
                            out_results_ptr->task_type = 0;
                            // Post the Full Results Object
                            svt_post_full_object(out_results_wrapper_ptr);
                        }

                        // Remove the Input Entry from the Input Queue
                        input_entry_ptr->input_object_ptr = (EbObjectWrapper *)NULL;
                    }
                }

                // Increment the head_index if the head is null
                encode_context_ptr->input_picture_queue_head_index =
                    (encode_context_ptr
                         ->input_picture_queue[encode_context_ptr->input_picture_queue_head_index]
                         ->input_object_ptr)
                        ? encode_context_ptr->input_picture_queue_head_index
                        : (encode_context_ptr->input_picture_queue_head_index ==
                           INPUT_QUEUE_MAX_DEPTH - 1)
                              ? 0
                              : encode_context_ptr->input_picture_queue_head_index + 1;

                // Increment the input_queue_index Iterator
                input_queue_index =
                    (input_queue_index == INPUT_QUEUE_MAX_DEPTH - 1) ? 0 : input_queue_index + 1;
            }

            // Walk the reference queue and remove entries that have been completely referenced.
            reference_queue_index = encode_context_ptr->reference_picture_queue_head_index;
            while (reference_queue_index !=
                   encode_context_ptr->reference_picture_queue_tail_index) {
                reference_entry_ptr =
                    encode_context_ptr->reference_picture_queue[reference_queue_index];

                // Remove the entry & release the reference if there are no remaining references
                if ((reference_entry_ptr->dependent_count == 0) &&
                    (reference_entry_ptr->reference_available) &&
                    (reference_entry_ptr->release_enable) &&
                    (reference_entry_ptr->reference_object_ptr)) {
                    // Release the nominal live_count value
                    svt_release_object(reference_entry_ptr->reference_object_ptr);
                    reference_entry_ptr->reference_object_ptr      = (EbObjectWrapper *)NULL;
                    reference_entry_ptr->reference_available       = EB_FALSE;
                    reference_entry_ptr->is_used_as_reference_flag = EB_FALSE;
                }

                // Increment the head_index if the head is empty
                encode_context_ptr->reference_picture_queue_head_index =
                    (encode_context_ptr
                         ->reference_picture_queue[encode_context_ptr
                                                       ->reference_picture_queue_head_index]
                         ->release_enable == EB_FALSE)
                        ? encode_context_ptr->reference_picture_queue_head_index
                        : (encode_context_ptr
                                   ->reference_picture_queue
                                       [encode_context_ptr->reference_picture_queue_head_index]
                                   ->reference_available == EB_FALSE &&
                           encode_context_ptr
                                   ->reference_picture_queue
                                       [encode_context_ptr->reference_picture_queue_head_index]
                                   ->is_used_as_reference_flag == EB_TRUE)
                              ? encode_context_ptr->reference_picture_queue_head_index
                              : (encode_context_ptr
                                     ->reference_picture_queue
                                         [encode_context_ptr->reference_picture_queue_head_index]
                                     ->dependent_count > 0)
                                    ? encode_context_ptr->reference_picture_queue_head_index
                                    : (encode_context_ptr->reference_picture_queue_head_index ==
                                       REFERENCE_QUEUE_MAX_DEPTH - 1)
                                          ? 0
                                          : encode_context_ptr->reference_picture_queue_head_index +
                                                1;
                // Increment the reference_queue_index Iterator
                reference_queue_index = (reference_queue_index == REFERENCE_QUEUE_MAX_DEPTH - 1)
                                            ? 0
                                            : reference_queue_index + 1;
            }
        }

        // Release the Input Picture Demux Results
        svt_release_object(input_picture_demux_wrapper_ptr);
    }
    return NULL;
}

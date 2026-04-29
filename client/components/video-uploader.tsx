"use client"

import { createClient } from "@/lib/supabase/client"
import { API_ENDPOINTS } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { useCallback, useRef, useState } from "react"
import { mutate } from "swr"

type AnalysisStatus = "idle" | "uploading" | "analyzing" | "done"

export function VideoUploader({ userId }: { userId: string }) {
  const [isDragging, setIsDragging] = useState(false)
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>("idle")
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [activeFileName, setActiveFileName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const isUploading = analysisStatus !== "idle" && analysisStatus !== "done"

  const handleUpload = useCallback(
    async (file: File) => {
      if (isUploading) {
        return
      }

      if (!file.type.startsWith("video/")) {
        setUploadError("Please upload a video file")
        return
      }

      if (file.size > 100 * 1024 * 1024) {
        setUploadError("File must be under 100 MB")
        return
      }

      setAnalysisStatus("uploading")
      setUploadError(null)
      setActiveFileName(file.name)

      const supabase = createClient()
      const fileExt = file.name.split(".").pop()
      const filePath = `${userId}/${Date.now()}.${fileExt}`

      // 1. Upload video to Supabase Storage
      const { error: storageError } = await supabase.storage
        .from("videos")
        .upload(filePath, file)

      if (storageError) {
        setUploadError("Upload failed. Please try again.")
        setAnalysisStatus("idle")
        setActiveFileName(null)
        return
      }

      const {
        data: { publicUrl },
      } = supabase.storage.from("videos").getPublicUrl(filePath)

      // 2. Insert a "processing" record so it shows immediately
      const { data: insertedRow, error: dbError } = await supabase
        .from("videos")
        .insert({
          user_id: userId,
          file_name: file.name,
          file_url: publicUrl,
          status: "processing",
        })
        .select("id")
        .single()

      if (dbError || !insertedRow) {
        setUploadError("Failed to save video record. Please try again.")
        setAnalysisStatus("idle")
        setActiveFileName(null)
        return
      }

      mutate("videos")

      // 3. Send video to backend for real AI analysis
      setAnalysisStatus("analyzing")

      try {
        const formData = new FormData()
        formData.append("file", file)

        const res = await fetch(API_ENDPOINTS.analyzeVideo, {
          method: "POST",
          body: formData,
        })

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Analysis failed" }))
          throw new Error(err.detail || "Analysis failed")
        }

        const analysis = await res.json()

        // 4. Update the DB record with real results
        if (analysis.status === "analyzed") {
          await supabase
            .from("videos")
            .update({
              status: "analyzed",
              shot_score: analysis.shot_score,
              arc_angle: analysis.arc_angle,
              release_speed: analysis.release_speed,
              follow_through_score: analysis.follow_through_score,
              llm_feedback: analysis.llm_feedback,
              shot_data: analysis.shot_data,
            })
            .eq("id", insertedRow.id)
        } else {
          // No shots detected
          await supabase
            .from("videos")
            .update({
              status: "analyzed",
              llm_feedback: analysis.message || "No shots detected in this video.",
            })
            .eq("id", insertedRow.id)
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Analysis failed"
        await supabase
          .from("videos")
          .update({
            status: "error",
            llm_feedback: `Analysis error: ${message}`,
          })
          .eq("id", insertedRow.id)
        setUploadError(message)
      }

      setAnalysisStatus("idle")
      setActiveFileName(null)
      if (fileInputRef.current) {
        // Allow re-selecting the same file after an attempt completes.
        fileInputRef.current.value = ""
      }
      mutate("videos")
    },
    [isUploading, userId]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleUpload(file)
    },
    [handleUpload]
  )

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleUpload(file)
    },
    [handleUpload]
  )

  return (
    <div>
      <div
        onDragOver={(e) => {
          if (isUploading) return
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => {
          if (isUploading) return
          setIsDragging(false)
        }}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 text-center transition-colors ${
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/40"
        }`}
      >
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-6 w-6 text-primary"
            aria-hidden="true"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        {isUploading ? (
          <>
            <p className="text-sm font-medium text-foreground">
              {analysisStatus === "uploading" && "Uploading video..."}
              {analysisStatus === "analyzing" && "Analyzing your shot with AI... This may take a moment."}
            </p>
            {activeFileName && (
              <p className="mt-2 text-xs text-muted-foreground">{activeFileName}</p>
            )}
          </>
        ) : (
          <>
            <p className="text-sm font-medium text-foreground">
              Drag and drop your video here
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              or click below to browse. Max 100 MB.
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              Choose file
            </Button>
          </>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="sr-only"
          onChange={handleFileChange}
          aria-label="Upload video file"
        />
      </div>

      {uploadError && (
        <p className="mt-3 text-center text-sm text-destructive">
          {uploadError}
        </p>
      )}
    </div>
  )
}
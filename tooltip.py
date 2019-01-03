import tkinter as tk


# creates a tooltip for the control panel buttons.
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 1000
        self.wraplength = 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        temp_id = self.id
        self.id = None
        if temp_id:
            self.widget.after_cancel(temp_id)

    # displays tooltip with info if user hovers over button
    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() - len(self.text)
        y += self.widget.winfo_rooty() + 40

        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)

        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#EEEEEE", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    # hide tip if mouse leaves
    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

//
//  ViewController.swift
//  ios-cpu-tool
//
//  Created by vvid on 12/10/2017.
//  Copyright © 2017 vvid. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var textOutput: UITextView!
    @IBOutlet weak var buttonStartStop: UIButton!

    var started: Bool = false

      @IBAction func startFreqCalc(_ sender: UIButton) {
        if (started)
        {
          started = false
          stop_thread()
          sender.setTitle("Start", for: .normal)
        }
        else
        {
          sender.setTitle("Stop", for: .normal)
          start_thread()
          started = true
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        textOutput.text = "Logs are dumped to stdout"
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

